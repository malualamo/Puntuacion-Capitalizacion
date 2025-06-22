import numpy as np
import torch
from sklearn.metrics import classification_report

PUNCT_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3, "¿": 4}
CAP_TAGS = {"lower": 0, "init": 1, "mix": 2, "upper": 3}


def train(model, dataloader_train, dataloader_test, optimizer, criterion_punct, criterion_cap, device, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, punct_labels, cap_labels in dataloader_train:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            punct_labels = punct_labels.to(device)
            cap_labels = cap_labels.to(device)

            optimizer.zero_grad()

            punct_logits, cap_logits = model(input_ids, attention_mask)

            loss_punct = criterion_punct(punct_logits.view(-1, punct_logits.shape[-1]), punct_labels.view(-1))
            loss_cap = criterion_cap(cap_logits.view(-1, cap_logits.shape[-1]), cap_labels.view(-1))

            loss = loss_punct + loss_cap
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader_train)

        """
        # Evaluación en test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, punct_labels, cap_labels in dataloader_test:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                punct_labels = punct_labels.to(device)
                cap_labels = cap_labels.to(device)

                punct_logits, cap_logits = model(input_ids, attention_mask)

                loss_punct = criterion(punct_logits.view(-1, punct_logits.shape[-1]), punct_labels.view(-1))
                loss_cap = criterion(cap_logits.view(-1, cap_logits.shape[-1]), cap_labels.view(-1))

                loss = loss_punct + loss_cap
                test_loss += loss.item()

        avg_test_loss = test_loss / len(dataloader_test)
        """

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")



def evaluate(model, dataloader, device):
    model.eval()

    inv_punct = {v: k for k, v in PUNCT_TAGS.items()}
    inv_cap = {v: k for k, v in CAP_TAGS.items()}

    all_true_punct = []
    all_pred_punct = []
    all_true_cap = []
    all_pred_cap = []



    with torch.no_grad():
        for input_ids, attention_mask, punct_labels, cap_labels in dataloader:
            input_ids = input_ids.to(device)
            punct_labels = punct_labels.to(device)
            cap_labels = cap_labels.to(device)

            punct_logits, cap_logits = model(input_ids)

            pred_punct = torch.argmax(punct_logits, dim=-1)
            pred_cap = torch.argmax(cap_logits, dim=-1)

            # Máscara para ignorar tokens sin etiqueta
            mask = (punct_labels != -100)

            # Aplicar máscara y aplanar para comparación
            all_true_punct.extend(punct_labels[mask].cpu().numpy())
            all_pred_punct.extend(pred_punct[mask].cpu().numpy())

            all_true_cap.extend(cap_labels[mask].cpu().numpy())
            all_pred_cap.extend(pred_cap[mask].cpu().numpy())

    print("Unique classes in true labels:", set(all_true_cap))
    print("Unique classes in predictions:", set(all_pred_cap))

    print("🔍 Unique true cap labels:", set(all_true_cap))
    print("🔍 Unique pred cap labels:", set(all_pred_cap))
    print("🔍 Unique true punct labels:", set(all_true_punct))
    print("🔍 Unique pred punct labels:", set(all_pred_punct))

    # Accuracy generales
    punct_acc = np.mean(np.array(all_true_punct) == np.array(all_pred_punct))
    cap_acc = np.mean(np.array(all_true_cap) == np.array(all_pred_cap))

    print("📌 Punctuation Accuracy:     {:.4f}".format(punct_acc))
    print("🔡 Capitalization Accuracy: {:.4f}".format(cap_acc))

    # Reportes detallados
    print("\n📊 Punctuation classification report:")
    print(classification_report(all_true_punct, all_pred_punct, target_names=[inv_punct[i] for i in range(len(inv_punct))]))

    print("\n📊 Capitalization classification report:")
    print(classification_report(all_true_cap, all_pred_cap, target_names=[inv_cap[i] for i in range(len(inv_cap))]))

    return punct_acc, cap_acc

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
