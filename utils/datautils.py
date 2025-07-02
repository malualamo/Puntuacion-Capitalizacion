
from collections import defaultdict
import json
import random
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
import wikipedia

from data.variables import *

def tiene_acento(word):
    word = re.sub(r"ñ", "n", word).lower()
    word = re.sub(r"#", "", word)
    cleaned_word = word.encode("ascii", "ignore").decode("utf-8")
    return word != cleaned_word


def get_capitalization_type(word):
    if not word or word.islower(): return 0
    if word.istitle(): return 1
    if word.isupper(): return 3
    if any(c.isupper() for c in word[1:]): return 2
    return 0

def get_cap_labels_for_tokens(labels_per_word, token_word_map):
    labels = []
    for word_idx in token_word_map:
        if word_idx is None:
            labels.append(-100)
        else:
            labels.append(labels_per_word[word_idx])
    return labels

def get_punct_start_labels_for_tokens(labels_per_word, token_word_map):
    labels = [0] * len(token_word_map)
    word_to_token_idxs = {}
    for token_idx, word_idx in enumerate(token_word_map):
        if word_idx is not None:
            word_to_token_idxs.setdefault(word_idx, []).append(token_idx)
    for word_idx, token_idxs in word_to_token_idxs.items():
        punct_label = labels_per_word[word_idx]
        labels[token_idxs[0]] = punct_label
    return labels


def get_punct_end_labels_for_tokens(labels_per_word, token_word_map):
    labels = [0] * len(token_word_map)
    word_to_token_idxs = {}
    for token_idx, word_idx in enumerate(token_word_map):
        if word_idx is not None:
            word_to_token_idxs.setdefault(word_idx, []).append(token_idx)
    for word_idx, token_idxs in word_to_token_idxs.items():
        punct_label = labels_per_word[word_idx]
        labels[token_idxs[-1]] = punct_label
    return labels


def get_dataloader(oraciones_raw, max_length, batch_size, device, tokenizer):
    input_ids_list = []
    attention_masks = []
    punct_start_labels_list = []
    punct_end_labels_list = []
    cap_labels_list = []

    for sent in oraciones_raw:
        # Extraer palabras con puntuación
        matches = list(re.finditer(r"\b\w+[^\s\w]?\b", sent)) # Detecta puntuaciones y las splitea
        words = []
        punct_start_labels = []
        punct_end_labels = []
        cap_labels = []

        for i, m in enumerate(matches): # Recorre cada palabra detectada
            word_raw = m.group(0) 
            clean_word = re.sub(r"[.,?¿]", "", word_raw) # Limpia la palabra "Hola!" -> "Hola"

            # Puntuación
            before = sent[m.start() - 1] if m.start() > 0 else "" # Signo anterior
            after = sent[m.end()] if m.end() < len(sent) else ""  # Signo posterior
            # Puntuación inicial (signo antes)
            if before in PUNCT_START_TAGS:
                punct_start = PUNCT_START_TAGS[before]
            else:
                punct_start = PUNCT_START_TAGS["Ø"]

            # Puntuación final (signo después)
            if after in PUNCT_END_TAGS:
                punct_end = PUNCT_END_TAGS[after]
            else:
                punct_end = PUNCT_END_TAGS["Ø"]

            # Capitalización
            cap = get_capitalization_type(word_raw)

            clean_word = clean_word.lower() # Limpia la palabra Hola -> hola

            words.append(clean_word)
            punct_start_labels.append(punct_start)
            punct_end_labels.append(punct_end)
            cap_labels.append(cap)

        # Tokenización con BERT
        encoding = tokenizer(words,
                             is_split_into_words=True,
                             return_tensors='pt',
                             padding='max_length',
                             truncation=True,
                             max_length=max_length,
                             return_attention_mask=True)

        # Extraer datos que nos sirven del encoding
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        word_ids = encoding.word_ids(batch_index=0)  # Mapea cada subtoken a su palabra

        # Alinear etiquetas a subtokens (hasta ahora las teniamos en palabras)
        punct_start_labels_aligned = get_punct_start_labels_for_tokens(punct_start_labels, word_ids)
        punct_end_labels_aligned = get_punct_end_labels_for_tokens(punct_end_labels, word_ids)
        cap_labels_aligned = get_cap_labels_for_tokens(cap_labels, word_ids)

        # Convertir a tensores
        punct_start_tensor = torch.tensor(punct_start_labels_aligned)
        punct_end_tensor = torch.tensor(punct_end_labels_aligned)
        cap_tensor = torch.tensor(cap_labels_aligned)

        # Aplicar -100 a posiciones de padding
        punct_start_tensor[attention_mask == 0] = -100
        punct_end_tensor[attention_mask == 0] = -100
        cap_tensor[attention_mask == 0] = -100

        # Agregar a listas (por oracion)
        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        punct_start_labels_list.append(punct_start_tensor)
        punct_end_labels_list.append(punct_end_tensor)
        cap_labels_list.append(cap_tensor)

    # Stackear tensores (por batch)
    input_ids        = torch.stack(input_ids_list).to(device)
    attention_masks  = torch.stack(attention_masks).to(device)
    punct_start_labels = torch.stack(punct_start_labels_list).to(device)
    punct_end_labels   = torch.stack(punct_end_labels_list).to(device)
    cap_labels         = torch.stack(cap_labels_list).to(device)

    dataset = TensorDataset(input_ids, attention_masks, punct_start_labels, punct_end_labels, cap_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def extraer_frases_dialogo(path_txt):
    personajes = [
    "Jorge", "Susana", "Mamá Cora", "Sergio", "Elvira", "Antonio", "Nora",
    "Matilde", "Dominga", "Felipe", "Emilia", "Doña Elisa", "Doña Gertrudis",
    "Don Genaro", "La Sorda", "Peralta", "Cacho", "Nene Florista"
    ]

    with open(path_txt, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = re.sub(r"\n+", "\n", raw)

    nombres_pattern = "|".join(re.escape(p) for p in personajes)
    pattern = re.compile(rf"^({nombres_pattern})\s*[\.:–\-]+", re.MULTILINE)

    frases = []
    matches = list(pattern.finditer(raw))

    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        bloque = raw[start:end]

        bloque = re.sub(r"\([^)]*\)", "", bloque)
        bloque = bloque.replace("\n", " ")
        bloque = re.sub(r"\s+", " ", bloque).strip()

        if bloque:
            frases.append(bloque)

    print(f"Se extrajeron {len(frases)} frases completas y se guardaron en 'dialogos_esperando_la_carroza.json'")
    return frases

def limpiar_simbolos(frases):
    frases_limpias = []
    for frase in frases:
        frase = re.sub(r"(\.\s*){2,}", "", frase)
        frase = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9¿?,. ]+", "", frase)
        frase = re.sub(r"\s+", " ", frase).strip()
        frases_limpias.append(frase)
    return frases_limpias

def cargar_json_wikipedia(temas):
    frases_wikipedia = []
    for tema in temas:
        print(f"Obteniendo frases de Wikipedia para: {tema}")
        frases = obtener_frases_wikipedia(tema,max_frases=100)
        frases_wikipedia.extend(frases)

    with open("data/frases_wikipedia.json", "w", encoding="utf-8") as f:
        json.dump(frases_wikipedia, f, ensure_ascii=False, indent=2)

    print("Frases guardadas en 'data/frases_wikipedia.json'")

def obtener_frases_wikipedia(titulo, max_frases=100):
    try:
        pagina = wikipedia.page(titulo)
        texto = pagina.content
        oraciones = re.split(r'(?<=[.!?])\s+', texto)
        frases = [o.strip() for o in oraciones if 5 < len(o.split()) < 30]
        return frases[:max_frases]
    except Exception as e:
        print(f"Error al buscar '{titulo}': {e}")
        return []
    
def extraer_frases_subtitulos(path_txt):
    with open(path_txt, "r", encoding="utf-8") as f:
        raw = f.read()

    bloques = re.split(r"\n\s*\n", raw)
    frases_crudas = []

    for bloque in bloques:
        lineas = bloque.strip().split("\n")
        if len(lineas) < 3:
            continue

        texto = " ".join(lineas[2:])
        texto = re.sub(r"^-", "", texto).strip()
        texto = re.sub(r"\s*-\s*", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()

        if not texto or re.fullmatch(r"[. ]{2,}", texto):
            continue

        frases_crudas.append(texto)

    frases_limpias = []
    buffer = ""

    for frase in frases_crudas:
        if buffer:
            buffer += " " + frase
        else:
            buffer = frase

        if re.search(r"[.!?](['”\"])?$", buffer):
            frases_limpias.append(buffer.strip())
            buffer = ""

    if buffer:
        frases_limpias.append(buffer.strip())

    print(f"✅ Se extrajeron {len(frases_limpias)} frases de Relatos Salvajes.")
    return frases_limpias

def predict_and_reconstruct(model, sentence, tokenizer, device, max_length=64, verbose=True):
    model.eval()

    encoding = tokenizer(
        sentence,
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
        # 🔄 Ahora el modelo devuelve 3 salidas: puntuación inicial, final y capitalización
        punct_start_logits, punct_end_logits, cap_logits = model(input_ids, attention_mask=attention_mask)

    # 🔍 Tomamos la predicción más probable (argmax) para cada token
    pred_punct_start = torch.argmax(punct_start_logits, dim=-1)[0].cpu().tolist()
    pred_punct_end   = torch.argmax(punct_end_logits, dim=-1)[0].cpu().tolist()
    pred_cap         = torch.argmax(cap_logits, dim=-1)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    INV_PUNCT_START_TAGS = {v: k for k, v in PUNCT_START_TAGS.items()}
    INV_PUNCT_END_TAGS   = {v: k for k, v in PUNCT_END_TAGS.items()}

    final_words = []
    current_word = ""
    current_cap = 0
    current_punct_start = 0
    current_punct_end = 0

    if verbose == True:
        print("\n🔍 Predicción token por token:")
        print(f"{'TOKEN':15s} | {'P_START':>7s} | {'P_END':>5s} | {'CAP':>3s} | {'FINAL':15s}")
        print("-" * 65)

    for i, (token, punct_start, punct_end, cap_label) in enumerate(zip(tokens, pred_punct_start, pred_punct_end, pred_cap)):
        if token in ["[CLS]", "[SEP]", "[PAD]"] or attention_mask[0, i].item() == 0:
            continue

        clean_token = token.replace("##", "")

        if token.startswith("##"):
            current_word += clean_token
            if punct_end != 0:
                current_punct_end = punct_end  # usar puntuación final del último subtoken relevante
        else:
            if current_word:
                # cerrar palabra anterior
                word = current_word
                # aplicar capitalización a toda la palabra
                if current_cap == 1:
                    word = word.capitalize()
                elif current_cap == 2:
                    word = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in word)
                elif current_cap == 3:
                    word = word.upper()

                # aplicar puntuación inicial
                punct_ini = INV_PUNCT_START_TAGS.get(current_punct_start, "Ø")
                if punct_ini == "¿":
                    word = "¿" + word
                elif punct_ini != "Ø":
                    word = punct_ini + word

                # aplicar puntuación final
                punct_fin = INV_PUNCT_END_TAGS.get(current_punct_end, "Ø")
                if punct_fin != "Ø":
                    word = word + punct_fin

                final_words.append(word)

            # empezar nueva palabra
            current_word = clean_token
            current_cap = cap_label
            current_punct_start = punct_start if punct_start != 0 else 0
            current_punct_end   = punct_end if punct_end != 0 else 0

        if verbose:
            print(f"{clean_token:15s} | {INV_PUNCT_START_TAGS.get(punct_start, 'Ø'):>7s} | {INV_PUNCT_END_TAGS.get(punct_end, 'Ø'):>5s} | {cap_label:3d} | {clean_token:15s}")

    # Procesar última palabra
    if current_word:
        word = current_word
        if current_cap == 1:
            word = word.capitalize()
        elif current_cap == 2:
            word = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in word)
        elif current_cap == 3:
            word = word.upper()
        punct_ini = INV_PUNCT_START_TAGS.get(current_punct_start, "Ø")
        if punct_ini == "¿":
            word = "¿" + word
        elif punct_ini != "Ø":
            word = punct_ini + word
        punct_fin = INV_PUNCT_END_TAGS.get(current_punct_end, "Ø")
        if punct_fin != "Ø":
            word = word + punct_fin
        final_words.append(word)

    return " ".join(final_words)

def procesar_oracion(sentence: str, tokenizer):
    matches = list(re.finditer(r"\b(\w|' )+[^\s\w]?\b", sentence, flags=re.UNICODE))
    original_words = [m.group(0) for m in matches]
    cleaned_words = [re.sub(r"[^A-Za-zÀ-ÿ]", "", w).lower() for w in original_words]

    encoding = tokenizer(
        cleaned_words,
        is_split_into_words=True,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    word_ids = encoding.word_ids(batch_index=0)

    word_to_token_idxs = defaultdict(list)
    for idx, wid in enumerate(word_ids):
        if wid is not None:
            word_to_token_idxs[wid].append(idx)

    total = len(cleaned_words)
    output = []
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue

        token = tokens[idx]
        prev_tok = tokens[idx-1] if idx > 0 else None
        next_tok = tokens[idx+1] if idx < len(tokens)-1 else None

        is_first = idx == word_to_token_idxs[wid][0]
        is_last = idx == word_to_token_idxs[wid][-1]

        has_accent = int(any(c in original_words[wid] for c in "áéíóúÁÉÍÓÚñÑüÜ"))
        pos_norm = round(wid / (total - 1), 2) if total > 1 else 0.0

        lead_char = sentence[matches[wid].start() - 1] if matches[wid].start() > 0 else 'Ø'
        start_punc = PUNCT_TAGS.get(lead_char, 0) if is_first else 0

        trail_char = sentence[matches[wid].end()] if matches[wid].end() < len(sentence) else 'Ø'
        end_punc = PUNCT_TAGS.get(trail_char, 0) if is_last else 0
        cap_type = get_capitalization_type(original_words[wid].strip("¿?.,"))

        output.append({
            "word": cleaned_words[wid],
            "token": tokenizer.convert_tokens_to_ids(token),
            "prev_token": tokenizer.convert_tokens_to_ids(prev_tok) if prev_tok and prev_tok != '[CLS]' else -1,
            "next_token": tokenizer.convert_tokens_to_ids(next_tok) if next_tok and next_tok != '[SEP]' else -1,
            "has_accent": has_accent,
            "position": pos_norm,
            "starting_punctuation_type": start_punc,
            "ending_punctuation_type": end_punc,
            "capitalization_type": cap_type,
        })

    return output
