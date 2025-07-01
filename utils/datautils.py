
from collections import Counter, defaultdict
import json
import random
import re
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import wikipedia

PUNCT_END_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3}
PUNCT_START_TAGS = {"Ø": 0, "¿": 4}
CAP_TAGS = {"lower": 0, "init": 1, "mix": 2, "upper": 3}

def tiene_acento(word):
    word = re.sub(r"ñ", "n", word).lower()
    word = re.sub(r"#", "", word)
    cleaned_word = word.encode("ascii", "ignore").decode("utf-8")
    return word != cleaned_word


def _get_capitalization_type(word):
    if not word or word.islower(): return 0
    if word.istitle(): return 1
    if word.isupper(): return 3
    if any(c.isupper() for c in word[1:]): return 2
    return 0

def get_cap_labels_for_tokens(labels_per_word, token_word_map):
    """
    Recibe los labels por palabra y devuelve los labels por token para capitalizacion
    Si los subtokens pertenecen a la misma palabra, les pone el mismo label (capitalizacion) 
    """
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
        # Puntuacion inicial va en primer subtoken
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
        # Puntuacion final va en último subtoken
        labels[token_idxs[-1]] = punct_label
    return labels


def get_dataloader(oraciones_raw, max_length, batch_size, device, tokenizer):
    """
    Crea un DataLoader para entrenar un modelo de restauración de puntuación y capitalización.

    A partir de una lista de oraciones correctamente escritas (con puntuación y mayúsculas),
    esta función:
        - Extrae etiquetas de puntuación y capitalización por palabra.
        - "Corrompe" el texto al eliminar la puntuación y poner las palabras en minúscula.
        - Tokeniza las palabras corruptas usando un tokenizer BERT.
        - Alinea las etiquetas con los subtokens del tokenizer.
        - Crea tensores para las entradas (input_ids, attention_mask) y etiquetas (puntuación y capitalización).
        - Devuelve un DataLoader para entrenamiento en lotes.

    Parámetros:
        oraciones_raw (List[str]): Lista de oraciones correctamente formateadas.
        max_length (int): Longitud máxima de secuencia para truncar/padear.
        batch_size (int): Tamaño del batch.
        device (str): Dispositivo donde se cargarán los tensores ('cpu' o 'cuda').

    Retorna:
        DataLoader: DataLoader que entrega batches de (input_ids, attention_mask, punct_labels, cap_labels).
    """
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
            cap = _get_capitalization_type(word_raw)

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

    # Limpiar saltos múltiples y unificar espacios
    raw = re.sub(r"\n+", "\n", raw)

    # Construir patrón para encontrar encabezados de personaje
    nombres_pattern = "|".join(re.escape(p) for p in personajes)
    pattern = re.compile(rf"^({nombres_pattern})\s*[\.:–\-]+", re.MULTILINE)

    frases = []
    matches = list(pattern.finditer(raw))

    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        bloque = raw[start:end]

        # Limpiar texto: eliminar paréntesis, saltos de línea, espacios múltiples
        bloque = re.sub(r"\([^)]*\)", "", bloque)
        bloque = bloque.replace("\n", " ")
        bloque = re.sub(r"\s+", " ", bloque).strip()

        if bloque:
            frases.append(bloque)

    print(f"✅ Se extrajeron {len(frases)} frases completas y se guardaron en 'dialogos_esperando_la_carroza.json'")
    return frases

def limpiar_simbolos(frases):
    frases_limpias = []
    for frase in frases:
        # Eliminar cualquier secuencia de dos o más puntos (incluyendo con espacios): ... . .. etc.
        frase = re.sub(r"(\.\s*){2,}", "", frase)
        # Eliminar cualquier caracter que NO sea letra, número, espacio o los signos permitidos
        frase = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9¿?,. ]+", "", frase)
        # Reemplazar múltiples espacios por uno solo
        frase = re.sub(r"\s+", " ", frase).strip()
        frases_limpias.append(frase)
    return frases_limpias

def cargar_json_wikipedia(archivo,temas, max_frases=100):
    """
    Carga un archivo JSON con frases de Wikipedia.
    """
    frases_wikipedia = []
    for tema in temas:
        print(f"Obteniendo frases de Wikipedia para: {tema}")
        frases = obtener_frases_wikipedia(tema,max_frases=100)
        print('Ejemplos de frases obtenidas:')
        for f in frases[:2]:
            print(f"- {f}")
        frases_wikipedia.extend(frases)
    # Guardar en un archivo JSON
    with open("data/frases_wikipedia.json", "w", encoding="utf-8") as f:
        json.dump(frases_wikipedia, f, ensure_ascii=False, indent=2)

    print("✅ Frases guardadas en 'data/frases_wikipedia.json'")

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

        # ❌ Filtrar frases vacías o solo puntos (como "..." o ". . .")
        if not texto or re.fullmatch(r"[. ]{2,}", texto):
            continue

        frases_crudas.append(texto)

    # 🔁 Unir frases incompletas (que no terminan en . ! ?)
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



def random_forest_predict_and_reconstruct(model, sentence, tokenizer, device, max_length=64, verbose=True):
    encoding = tokenizer(
        sentence,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False
    )

    words = sentence.split()
    for word in words:
        tokens = tokenizer.tokenize(word)
        preds = []
        for token in tokens:

            print(f"Token: {token}")
            #if token.startswith("##"):
            #    continue
            # model is a scikit-learn rf model
            cap_pred = model.predict([tokenizer.convert_tokens_to_ids(token)])[0]
            preds.append(cap_pred)
        # Reemplazar el token original por el token con la capitalización predicha de mayor ocurrencia
        if preds:
            most_common_cap = max(set(preds), key=preds.count)
            if most_common_cap == 1:
                word = word.capitalize()
            elif most_common_cap == 2:
                word = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in word)
            elif most_common_cap == 3:
                word = word.upper()
        sentence = sentence.replace(word, word, 1)
    return sentence


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
        cap_type = _get_capitalization_type(original_words[wid].strip("¿?.,"))

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

ACCENT_RE = re.compile(r"[áéíóúÁÉÍÓÚñÑüÜ]")
DEFAULT_PUNCT_MAP = {0: "", 1: ",", 2: ".", 3: "?", 4: "¿"}
brands = {"mcdonald's": "McDonald's",}

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
