import torch
import random
PUNCT_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3, "¿": 4}
CAP_TAGS = {"lower": 0, "init": 1, "mix": 2, "upper": 3}

def predict_and_reconstruct(model, sentence, tokenizer, max_length=64):
    """
    Aplica inferencia al modelo y reconstruye el texto con puntuación y capitalización.
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenización
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        punct_logits, cap_logits = model(input_ids, attention_mask=attention_mask)

    # Decodificar predicciones
    pred_punct = torch.argmax(punct_logits, dim=-1)[0].cpu().tolist()
    pred_cap   = torch.argmax(cap_logits,   dim=-1)[0].cpu().tolist()
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])
    INV_PUNCT  = {v:k for k,v in PUNCT_TAGS.items()}

    final_words = []
    cur_word, cur_cap, cur_punct = "", 0, 0

    for i, tok in enumerate(tokens):
        if tok in ["[CLS]","[SEP]","[PAD]"] or attention_mask[0, i] == 0:
            continue
        clean = tok.replace("##", "")
        if tok.startswith("##"):
            cur_word += clean
            if pred_punct[i] != 0:
                cur_punct = pred_punct[i]
        else:
            # cerrar palabra anterior
            if cur_word:
                w = cur_word
                if cur_cap == 1:
                    w = w.capitalize()
                elif cur_cap == 2:
                    w = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in w)
                elif cur_cap == 3:
                    w = w.upper()
                p = INV_PUNCT.get(cur_punct, "Ø")
                if p == "¿":
                    w = "¿" + w
                elif p != "Ø":
                    w = w + p
                final_words.append(w)
            # nueva palabra
            cur_word, cur_cap, cur_punct = clean, pred_cap[i], pred_punct[i]

    # última palabra
    if cur_word:
        w = cur_word
        if cur_cap == 1:
            w = w.capitalize()
        elif cur_cap == 2:
            w = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in w)
        elif cur_cap == 3:
            w = w.upper()
        p = INV_PUNCT.get(cur_punct, "Ø")
        if p == "¿":
            w = "¿" + w
        elif p != "Ø":
            w = w + p
        final_words.append(w)

    return " ".join(final_words)
