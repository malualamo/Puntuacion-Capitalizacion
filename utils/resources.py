import random
import re
import unicodedata
from datasets import load_dataset
import json
import wikipedia
from utils.datautils import extraer_frases_dialogo, extraer_frases_subtitulos, limpiar_simbolos
from data.variables import *

def get_questions():
    DATA_URLS = {
    "train": "https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/train.json",
    "dev":   "https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/dev.json",
    "test":  "https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/test.json",
}

    raw = load_dataset(
        "json",
        data_files=DATA_URLS,
        field="data",
    )

    questions = []

    for i in range(0, len(raw["train"])):
        for p in raw["train"][i]['paragraphs']:
            p_questions = [qas['question'] for qas in p['qas']]
            questions += p_questions

    N_QUESTIONS = 5000
    raw_questions = questions[:N_QUESTIONS] 
    for_mixture = questions[N_QUESTIONS:N_QUESTIONS + 5000]
    print(f"Se descargaron {len(raw_questions)} preguntas en Español.")
    return raw_questions, for_mixture


def get_notebook_dataset():
    dataset_rnn = load_dataset("google/wmt24pp", "en-es_MX", split="train")
    oraciones_rnn = dataset_rnn['target'][1:]

    print(f"Se descargaron {len(oraciones_rnn)} oraciones en Español (del dataset del notebook 10).")
    return oraciones_rnn


def get_gemini_dataset():
    oraciones_sinteticas = []
    with open('./data/datasets.json', 'r',encoding='utf-8') as file:
        data = json.load(file)

    oraciones_sinteticas = data['otros'] + data['marcas']
    print(f"Hay {len(oraciones_sinteticas)} oraciones sintéticas.")
    return oraciones_sinteticas


def get_wikipedia_dataset():
    wikipedia.set_lang("es")

    with open("data/frases_wikipedia.json", "r", encoding="utf-8") as f:
        frases_wikipedia = json.load(f)

    print(f"Se cargaron {len(frases_wikipedia)} frases de Wikipedia.")

    return frases_wikipedia


def get_pelis_dataset():
    esperando_la_carroza = extraer_frases_dialogo("data/esperando_la_carroza.txt")

    with open("data/dialogos_esperando_la_carroza.json", "w", encoding="utf-8") as f:
        json.dump(esperando_la_carroza, f, ensure_ascii=False, indent=2)

    print("Frases extraídas en total:", len(esperando_la_carroza))

    frases_relatos_salvajes = extraer_frases_subtitulos("data/subt_relatos_salvajes.srt")

    with open("data/frases_relatos_salvajes.json", "w", encoding="utf-8") as f:
        json.dump(frases_relatos_salvajes, f, ensure_ascii=False, indent=2)

    frases_relatos_salvajes = limpiar_simbolos(frases_relatos_salvajes)
    esperando_la_carroza = limpiar_simbolos(esperando_la_carroza)

    return esperando_la_carroza, frases_relatos_salvajes

def get_mixture_dataset(oraciones_sinteticas, question_for_mixture):
    cant_oraciones = len(oraciones_sinteticas)
    question_for_mixture = [re.sub(r'[\\\(\)!¡“]', '', unicodedata.normalize("NFC", q).strip()) for q in question_for_mixture]
    oraciones_sinteticas = [re.sub(r'[\\\(\)!¡“]', '', unicodedata.normalize("NFC", a).strip()) for a in oraciones_sinteticas]

    tanda_1 = question_for_mixture[:cant_oraciones]
    question_affirmation = [f"{q} {a}" for q, a in zip(tanda_1, oraciones_sinteticas)]

    tanda_2 = question_for_mixture[cant_oraciones:2*cant_oraciones]
    affirmation_question = [f"{a} {q}" for q, a in zip(tanda_2, oraciones_sinteticas)]

    tanda_3 = question_for_mixture[2*cant_oraciones:3*cant_oraciones]
    tanda_3_shuffled = random.sample(tanda_3, len(tanda_3))
    question_question = [f"{q} {p}" for q, p in zip(tanda_3, tanda_3_shuffled)]

    mixtures = question_affirmation + affirmation_question + question_question
    return mixtures