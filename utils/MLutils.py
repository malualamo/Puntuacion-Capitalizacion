import numpy as np
import torch
from sklearn.metrics import classification_report

PUNCT_END_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3}
PUNCT_START_TAGS = {"Ø": 0, "¿": 1}
CAP_TAGS = {"lower": 0, "init": 1, "mix": 2, "upper": 3}

def train(model, dataloader_train, dataloader_test, optimizer,
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


from collections import Counter
import torch

def compute_class_weights(
    dataloader,
    num_punct_classes: int,
    num_cap_classes: int,
    device,
    beta: float = 0.7,
    ignore_index: int = -100,
    clamp_min: float = 1.0,
    clamp_max: float = 5.0
):
    """
    Recorre el dataloader para contar ocurrencias de cada etiqueta (ignorando ignore_index),
    calcula los pesos inversos elevados a beta, y devuelve dos tensores de peso
    para CrossEntropyLoss (puntuación y capitalización).
    """
    punct_counter = Counter()
    cap_counter   = Counter()

    # 1) contar etiquetas
    for _, _, punct_labels, cap_labels in dataloader:
        pl = punct_labels.cpu().numpy().ravel()
        cl = cap_labels.cpu().numpy().ravel()

        valid_p = pl[pl != ignore_index]
        valid_c = cl[cl != ignore_index]

        punct_counter.update(valid_p)
        cap_counter.update(valid_c)

    total_p = sum(punct_counter.values())
    total_c = sum(cap_counter.values())

    # 2) calcular pesos inversos^beta
    punct_weights = {
        tag: (total_p / count) ** beta
        for tag, count in punct_counter.items()
    }
    cap_weights = {
        tag: (total_c / count) ** beta
        for tag, count in cap_counter.items()
    }

    # 3) tensor con clamp
    punct_w_tensor = (
        torch.tensor(
            [punct_weights.get(i, 1.0) for i in range(num_punct_classes)],
            dtype=torch.float32
        )
        .to(device)
        .clamp(min=clamp_min, max=clamp_max)
    )
    cap_w_tensor = (
        torch.tensor(
            [cap_weights.get(i, 1.0) for i in range(num_cap_classes)],
            dtype=torch.float32
        )
        .to(device)
        .clamp(min=clamp_min, max=clamp_max)
    )

    return punct_w_tensor, cap_w_tensor
