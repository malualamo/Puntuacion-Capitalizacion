
import json
import random
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
import wikipedia

PUNCT_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3, "¿": 4}
CAP_TAGS = {"lower": 0, "init": 1, "mix": 2, "upper": 3}

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

def get_punct_labels_for_tokens(labels_per_word, token_word_map):
    """
    Asigna etiquetas de puntuación a los subtokens, siguiendo las reglas:
    - ¿ va en el primer subtoken de la palabra.
    - ., ?, , van en el último subtoken de la palabra.
    - Ø no se asigna a ningún subtoken (todos -100).
    """
    labels = [0] * len(token_word_map)
    word_to_token_idxs = {}

    # Construimos un diccionario: word_idx -> [lista de posiciones de tokens]
    for token_idx, word_idx in enumerate(token_word_map):
        if word_idx is not None:
            word_to_token_idxs.setdefault(word_idx, []).append(token_idx)

    for word_idx, token_idxs in word_to_token_idxs.items():
        punct_label = labels_per_word[word_idx]
        if punct_label == PUNCT_TAGS["¿"]:
            target_idx = token_idxs[0]  # primer subtoken
        elif punct_label in {PUNCT_TAGS["."], PUNCT_TAGS[","], PUNCT_TAGS["?"]}:
            target_idx = token_idxs[-1]  # último subtoken
        else:
            continue  # Ø: no se asigna nada

        labels[target_idx] = punct_label

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
    punct_labels_list = []
    cap_labels_list = []

    for sent in oraciones_raw:
        # Extraer palabras con puntuación
        matches = list(re.finditer(r"\b\w+[^\s\w]?\b", sent)) # Detecta puntuaciones y las splitea
        words = []
        punct_labels = []
        cap_labels = []

        for i, m in enumerate(matches): # Recorre cada palabra detectada
            word_raw = m.group(0) 
            clean_word = re.sub(r"[.,?¿]", "", word_raw) # Limpia la palabra "Hola!" -> "Hola"

            # Puntuación
            before = sent[m.start() - 1] if m.start() > 0 else "" # Signo anterior
            after = sent[m.end()] if m.end() < len(sent) else ""  # Signo posterior
            if before == '¿':
                punct = PUNCT_TAGS["¿"]
            elif after in PUNCT_TAGS:
                punct = PUNCT_TAGS[after]
            else:
                punct = PUNCT_TAGS["Ø"]

            # Capitalización
            cap = _get_capitalization_type(word_raw)

            clean_word = clean_word.lower() # Limpia la palabra Hola -> hola

            words.append(clean_word)
            punct_labels.append(punct)
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
        punct_labels_aligned = get_punct_labels_for_tokens(punct_labels, word_ids)
        cap_labels_aligned = get_cap_labels_for_tokens(cap_labels, word_ids)

        # Convertir a tensores
        punct_tensor = torch.tensor(punct_labels_aligned)
        cap_tensor = torch.tensor(cap_labels_aligned)

        # Aplicar -100 a posiciones de padding
        punct_tensor[attention_mask == 0] = -100
        cap_tensor[attention_mask == 0] = -100

        # Agregar a listas (por oracion)
        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        punct_labels_list.append(punct_tensor)
        cap_labels_list.append(cap_tensor)

    # Stackear tensores (por batch)
    input_ids = torch.stack(input_ids_list).to(device)
    attention_masks = torch.stack(attention_masks).to(device)
    punct_labels = torch.stack(punct_labels_list).to(device)
    cap_labels = torch.stack(cap_labels_list).to(device)

    dataset = TensorDataset(input_ids, attention_masks, punct_labels, cap_labels)
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
        punct_logits, cap_logits = model(input_ids, attention_mask=attention_mask)

    pred_punct = torch.argmax(punct_logits, dim=-1)[0].cpu().tolist()
    pred_cap = torch.argmax(cap_logits, dim=-1)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    INV_PUNCT_TAGS = {v: k for k, v in PUNCT_TAGS.items()}

    final_words = []
    current_word = ""
    current_cap = 0
    current_punct = 0
    new_word = True

    if verbose == True:
        print("\n🔍 Predicción token por token:")
        print(f"{'TOKEN':15s} | {'PUNCT':>5s} | {'SIGNO':>5s} | {'CAP':>3s} | {'FINAL':15s}")
        print("-" * 55)

    for i, (token, punct_label, cap_label) in enumerate(zip(tokens, pred_punct, pred_cap)):
        if token in ["[CLS]", "[SEP]", "[PAD]"] or attention_mask[0, i].item() == 0:
            continue

        clean_token = token.replace("##", "")

        if token.startswith("##"):
            current_word += clean_token
            if punct_label != 0:
                current_punct = punct_label  # usar puntuación del último subtoken relevante
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
                # aplicar puntuación del último subtoken
                punct = INV_PUNCT_TAGS.get(current_punct, "Ø")
                if punct == "¿":
                    word = "¿" + word
                elif punct != "Ø":
                    word = word + punct
                final_words.append(word)

            # empezar nueva palabra
            current_word = clean_token
            current_cap = cap_label
            current_punct = punct_label if punct_label != 0 else 0

        if verbose:
            print(f"{clean_token:15s} | {punct_label:5d} | {INV_PUNCT_TAGS.get(punct_label, 'Ø'):>5s} | {cap_label:3d} | {clean_token:15s}")

    # Procesar última palabra
    if current_word:
        word = current_word
        if current_cap == 1:
            word = word.capitalize()
        elif current_cap == 2:
            word = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in word)
        elif current_cap == 3:
            word = word.upper()
        punct = INV_PUNCT_TAGS.get(current_punct, "Ø")
        if punct == "¿":
            word = "¿" + word
        elif punct != "Ø":
            word = word + punct
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