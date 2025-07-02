import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from collections import Counter, defaultdict
from data import brands
from data.variables import *


def train(model, dataloader_train, optimizer,
          criterion_start, criterion_end, criterion_cap, device, epochs=3):
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, punct_start_labels, punct_end_labels, cap_labels in dataloader_train:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            punct_start_labels = punct_start_labels.to(device)
            punct_end_labels = punct_end_labels.to(device)
            cap_labels = cap_labels.to(device)

            optimizer.zero_grad()

            punct_start_logits, punct_end_logits, cap_logits = model(input_ids, attention_mask)

            loss_start = criterion_start(punct_start_logits.view(-1, punct_start_logits.shape[-1]),
                                         punct_start_labels.view(-1))
            loss_end = criterion_end(punct_end_logits.view(-1, punct_end_logits.shape[-1]),
                                     punct_end_labels.view(-1))
            loss_cap = criterion_cap(cap_logits.view(-1, cap_logits.shape[-1]),
                                     cap_labels.view(-1))

            loss = loss_start + loss_end + loss_cap
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader_train)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

def evaluate(model, dataloader, device):
    model.eval()

    inv_start = {v: k for k, v in PUNCT_START_TAGS.items()}
    inv_end   = {v: k for k, v in PUNCT_END_TAGS.items()}
    inv_cap   = {v: k for k, v in CAP_TAGS.items()}

    all_true_start, all_pred_start = [], []
    all_true_end,   all_pred_end   = [], []
    all_true_cap,   all_pred_cap   = [], []

    with torch.no_grad():
        for input_ids, attention_mask, punct_start_labels, punct_end_labels, cap_labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            punct_start_logits, punct_end_logits, cap_logits = model(input_ids, attention_mask)
            pred_start = punct_start_logits.argmax(dim=-1)
            pred_end   = punct_end_logits.argmax(dim=-1)
            pred_cap   = cap_logits.argmax(dim=-1)

            mask_start = (punct_start_labels != -100)
            mask_end   = (punct_end_labels   != -100)
            mask_cap   = (cap_labels         != -100)

            ts = punct_start_labels[mask_start].cpu().numpy()
            ps = pred_start[mask_start].cpu().numpy()
            all_true_start.extend(ts)
            all_pred_start.extend(ps)

            te = punct_end_labels[mask_end].cpu().numpy()
            pe = pred_end[mask_end].cpu().numpy()
            all_true_end.extend(te)
            all_pred_end.extend(pe)

            tc = cap_labels[mask_cap].cpu().numpy()
            pc = pred_cap[mask_cap].cpu().numpy()
            all_true_cap.extend(tc)
            all_pred_cap.extend(pc)

    print(f"Start Acc: {np.mean(np.array(all_true_start)==np.array(all_pred_start)):.4f}")
    print(f"End   Acc: {np.mean(np.array(all_true_end)==np.array(all_pred_end)):.4f}")
    print(f"Cap   Acc: {np.mean(np.array(all_true_cap)==np.array(all_pred_cap)):.4f}")

    print("\nStart report:")
    print(classification_report(all_true_start, all_pred_start,
                                target_names=[inv_start[i] for i in sorted(inv_start)]))
    print("\nEnd report:")
    print(classification_report(all_true_end, all_pred_end,
                                target_names=[inv_end[i]   for i in sorted(inv_end)]))
    print("\nCap report:")
    print(classification_report(all_true_cap, all_pred_cap,
                                target_names=[inv_cap[i]   for i in sorted(inv_cap)]))

def compute_class_weights(
    dataloader,
    num_classes_list: list[int],
    device,
    beta: float = 0.7,
    ignore_index: int = -100,
    clamp_min: float = 1.0,
    clamp_max: float = 5.0
) -> list[torch.Tensor]:
    n_tasks = len(num_classes_list)
    counters = [Counter() for _ in range(n_tasks)]

    for batch in dataloader:
        label_tensors = batch[-n_tasks:]
        for lbl_tensor, counter in zip(label_tensors, counters):
            arr = lbl_tensor.cpu().numpy().ravel()
            valid = arr[arr != ignore_index]
            counter.update(valid)

    weight_tensors = []
    for counter, num_cls in zip(counters, num_classes_list):
        total = sum(counter.values())
        weights = {cls: (total/count)**beta for cls, count in counter.items()}
        w_tensor = (
            torch.tensor([weights.get(i, 1.0) for i in range(num_cls)],
                         dtype=torch.float32)
            .to(device)
            .clamp(min=clamp_min, max=clamp_max)
        )
        weight_tensors.append(w_tensor)

    return weight_tensors

def predicciones_TP(txt_path, model, tokenizer, device, max_length=128, verbose=False):
    """
    Procesa un archivo TXT con párrafos (una instancia por párrafo).
    Devuelve un CSV con columnas:
    instancia_id, token_id, token, punt_inicial, punt_final, capitalización

    Parámetros:
    - txt_path: ruta al archivo TXT.
    - model: modelo entrenado con 3 cabezas.
    - tokenizer: tokenizer BERT.
    - device: 'cpu' o 'cuda'.
    - max_length: max tokens por instancia.
    - verbose: si imprime detalles.
    """

    model.eval()

    INV_PUNCT_START_TAGS = {v:k for k,v in PUNCT_START_TAGS.items()}
    INV_PUNCT_END_TAGS = {v:k for k,v in PUNCT_END_TAGS.items()}
    INV_CAP_TAGS = {v:k for k,v in CAP_TAGS.items()}

    results = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        paragraphs = [p.strip() for p in f.read().split('\n\n') if p.strip()]

    for instancia_id, paragraph in enumerate(paragraphs, start=1):
        encoding = tokenizer(
            paragraph,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=False
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            # Suponiendo que el modelo devuelve 3 logits (punct_start, punct_end, cap)
            punct_start_logits, punct_end_logits, cap_logits = model(input_ids, attention_mask=attention_mask)

        pred_punct_start = torch.argmax(punct_start_logits, dim=-1)[0].cpu().tolist()
        pred_punct_end = torch.argmax(punct_end_logits, dim=-1)[0].cpu().tolist()
        pred_cap = torch.argmax(cap_logits, dim=-1)[0].cpu().tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        if verbose:
            print(f"\nInstancia {instancia_id}: {paragraph}")
            print(f"{'Token ID':>8} | {'Token':15} | {'Punt Inicial':12} | {'Punt Final':10} | {'Capitalizacion':13}")
            print("-"*70)

        for token_id, (token, pstart, pend, cap) in enumerate(zip(tokens, pred_punct_start, pred_punct_end, pred_cap)):
            # Ignorar tokens especiales y padding
            if token in ["[CLS]", "[SEP]", "[PAD]"] or attention_mask[0, token_id].item() == 0:
                continue

            results.append({
                'instancia_id': instancia_id,
                'token_id': tokenizer.convert_tokens_to_ids(token),
                'token': token,
                'punt_inicial': INV_PUNCT_START_TAGS.get(pstart, ""),
                'punt_final': INV_PUNCT_END_TAGS.get(pend, ""),
                'capitalización': INV_CAP_TAGS.get(cap, "Ø")
            })

            if verbose:
                print(f"{token_id:8} | {token:15} | {INV_PUNCT_START_TAGS.get(pstart, 'Ø'):12} | {INV_PUNCT_END_TAGS.get(pend, 'Ø'):10} | {INV_CAP_TAGS.get(cap, 'Ø'):13}")

    df = pd.DataFrame(results)
    csv_path = txt_path.replace('.txt', '_predicciones.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    if verbose:
        print(f"\nPredicciones guardadas en: {csv_path}")

    return df

def random_forest_predict_and_reconstruct(
    cap_model,
    punct_start_model,
    punct_end_model,
    sentence,
    tokenizer,
    punct_map = None,
    verbose = False
):
    if punct_map is None:
        punct_map = DEFAULT_PUNCT_MAP

    flat_toks = tokenizer.tokenize(sentence)
    total = len(flat_toks)
    denom = total - 1 if total > 1 else 1
    ptr = 0
    rows = []
    instance_id = 1
    new_words = []
    for word in sentence.split():
        subtoks = tokenizer.tokenize(word)
        caps_preds = []
        start_pred = 0
        end_pred = 0

        for i in range(len(subtoks)):
            tok = flat_toks[ptr]
            if verbose:
                print("Token:", tok)
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            has_accent = 1 if ACCENT_RE.search(tok) else 0
            norm_pos = ptr / denom
            next_tok = tokenizer.convert_tokens_to_ids(flat_toks[ptr + 1]) if ptr + 1 < len(flat_toks) else -1
            prev_tok = tokenizer.convert_tokens_to_ids(flat_toks[ptr - 1]) if ptr > 0 else -1
            feats = [tok_id, prev_tok, next_tok, has_accent, norm_pos]
            caps_preds.append(cap_model.predict([feats])[0])
            if i == 0:
                start_pred = punct_start_model.predict([feats])[0]
            if i == len(subtoks) - 1:
                end_pred = punct_end_model.predict([feats])[0]
            ptr += 1

        counter = Counter(caps_preds)
        if caps_preds[0] == 1:
            cap_choice = 1
        elif counter.get(2, 0) > 1:
            cap_choice = 2
        else:
            cap_choice = counter.most_common(1)[0][0]

        if cap_choice == 1:
            mod = word.capitalize()
        elif cap_choice == 2:
            if word.lower() in brands:
                mod = brands[word.lower()]
            else:
                mod = "".join(
                    c.upper() if random.random() > 0.5 else c.lower()
                    for c in word
                )
        elif cap_choice == 3:
            mod = word.upper()
        else:
            mod = word

        prefix = punct_map.get(start_pred, "")
        suffix = punct_map.get(end_pred,   "")
        mod = f"{prefix}{mod}{suffix}"

        new_words.append(mod)

        for idx, tok in enumerate(subtoks):
            token_id = tokenizer.convert_tokens_to_ids(flat_toks[ptr - len(subtoks) + idx])
            rows.append({
                "instancia_id": instance_id,
                "token_id": token_id,
                "token": tok,
                "punt_inicial": start_pred if idx == 0 else 0,
                "punt_final": end_pred if idx == len(subtoks) - 1 else 0,
                "capitalización": cap_choice
            })
        instance_id += 1

    df = pd.DataFrame(rows)
    return df, " ".join(new_words)


def undersample(
    X,
    y,
    freq,
):
    idx_by_class = defaultdict(list)
    for i, label in enumerate(y):
        idx_by_class[label].append(i)

    selected_idxs = []
    rnd = random.Random(42)
    for cls, indices in idx_by_class.items():
        if cls in freq:
            desired = freq[cls]
            available = len(indices)
            if desired > available:
                desired = available
            chosen = rnd.sample(indices, desired)
        else:
            chosen = list(indices)
        selected_idxs.extend(chosen)

    rnd.shuffle(selected_idxs)
    X_res = [X[i] for i in selected_idxs]
    y_res = [y[i] for i in selected_idxs]
    return X_res, y_res

