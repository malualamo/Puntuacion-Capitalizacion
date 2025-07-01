import random
from datasets import load_dataset
import json

import wikipedia

from utils.datautils import extraer_frases_dialogo, extraer_frases_subtitulos, limpiar_simbolos

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
    with open('./data/datasets.json', 'r') as file:
        data = json.load(file)

    oraciones_sinteticas = data['otros'] + data['marcas']
    print(f"Hay {len(oraciones_sinteticas)} oraciones sintéticas.")
    return oraciones_sinteticas


def get_wikipedia_dataset():
    wikipedia.set_lang("es")

    temas = [
        # Países y lugares
        'Argentina', 'España', 'México', 'Colombia', 'Chile',
        'Perú', 'Uruguay', 'Brasil', 'América Latina', 'Europa',

        # Cultura argentina
        'Lionel Messi', 'Diego Maradona', 'Lali Esposito', 'Charly Garcia', 'Dillom',
        'Tiempos Violentos', 'Relatos Salvajes', 'Universidad de Buenos Aires', 'Rock nacional', 'Cine argentino',

        # Historia y política
        'Revolucion de Mayo', 'Independencia de Argentina', 'Simón Bolívar', 'Segunda Guerra Mundial', 'Guerra Fría',
        'Revolución Francesa', 'Guerra Civil Española', 'Napoleón Bonaparte', 'Nelson Mandela', 'Dictadura militar en Argentina',

        # Ciencia y tecnología
        'Inteligencia artificial', 'ChatGPT', 'Redes neuronales', 'Robótica', 'Energía solar',
        'Vacunas', 'COVID-19', 'Cambio climático', 'Computadora cuántica', 'NASA',

        # Cultura general
        'El Principito', 'Premio Nobel', 'Frida Kahlo', 'Pablo Picasso', 'Leonardo da Vinci',
        'William Shakespeare', 'Gabriel García Márquez', 'Julio Cortázar', 'Literatura latinoamericana', 'Arte contemporáneo',

        # Entretenimiento y medios
        'Marvel', 'DC Comics', 'Netflix', 'Cine de terror', 'Películas de ciencia ficción',
        'Música electrónica', 'Reguetón', 'Spotify', 'YouTube', 'TikTok',

        # Deportes
        'Fútbol', 'Copa Mundial de la FIFA', 'Juegos Olimpicos', 'Tenis', 'NBA',
        'Boca Juniors', 'River Plate', 'Messi vs Ronaldo', 'Fórmula 1', 'Michael Jordan',

        # Sociedad y actualidad
        'Feminismo', 'Día Internacional de la Mujer', 'Diversidad cultural', 'Migración', 'Pobreza',
        'Educación pública', 'Salud mental', 'Medio ambiente', 'Derechos humanos', 'Trabajo remoto',

        # Filosofía y pensamiento
        'Filosofía', 'Ética', 'Psicología', 'Sigmund Freud', 'Carl Jung',
        'Existencialismo', 'Sociología', 'Economía', 'Política', 'Democracia'
    ]


    with open("data/frases_wikipedia.json", "r", encoding="utf-8") as f:
        frases_wikipedia = json.load(f)

    print(frases_wikipedia[:5])  # muestra las primeras frases

    return frases_wikipedia


def get_pelis_dataset():
    esperando_la_carroza = extraer_frases_dialogo("data/esperando_la_carroza.txt")

    with open("data/dialogos_esperando_la_carroza.json", "w", encoding="utf-8") as f:
        json.dump(esperando_la_carroza, f, ensure_ascii=False, indent=2)

    print("✅ Frases extraídas y guardadas. Total:", len(esperando_la_carroza))
    print(random.sample(esperando_la_carroza, 10))

    frases_relatos_salvajes = extraer_frases_subtitulos("data/subt_relatos_salvajes.srt")

    # Guardar como JSON
    with open("data/frases_relatos_salvajes.json", "w", encoding="utf-8") as f:
        json.dump(frases_relatos_salvajes, f, ensure_ascii=False, indent=2)

    frases_relatos_salvajes = limpiar_simbolos(frases_relatos_salvajes)
    esperando_la_carroza = limpiar_simbolos(esperando_la_carroza)

    return esperando_la_carroza, frases_relatos_salvajes