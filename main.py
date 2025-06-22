import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import torch
from transformers import BertModel, BertTokenizerFast
from huggingface_hub import hf_hub_download
from model import PunctuationCapitalizationRNN, PunctuationCapitalizationRNNBidirectional, PUNCT_TAGS, CAP_TAGS
from utils import predict_and_reconstruct

try:
    from google import genai
    from google.genai import types
except ImportError:
    import google.generativeai as genai
    from google.generativeai import types

client = genai.Client(
    api_key=st.secrets.get("GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY"),
)

MODEL_REPOS = {
    "RNN clásica (última versión)": "fabroo/PunctuationCapitalizationRNN",
    "RNN bidireccional (última versión)": "fabroo/PunctuationCapitalizationRNN",
}

@st.cache_resource
def load_classic(repo_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained("bert-base-multilingual-cased")
    for p in bert.embeddings.word_embeddings.parameters(): p.requires_grad = False
    for layer in bert.encoder.layer[-2:]:
        for p in layer.parameters(): p.requires_grad = True
    for p in bert.pooler.parameters(): p.requires_grad = True
    model = PunctuationCapitalizationRNN(
        bert_model=bert,
        hidden_dim=256,
        num_punct_classes=len(PUNCT_TAGS),
        num_cap_classes=len(CAP_TAGS),
        dropout=0.3
    ).to(device)
    state_path = hf_hub_download(repo_id=repo_id, filename="modelo_fine_tuned_state_dict.pt")
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_bidirectional(repo_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained("bert-base-multilingual-cased")
    for p in bert.embeddings.word_embeddings.parameters(): p.requires_grad = False
    for layer in bert.encoder.layer[-2:]:
        for p in layer.parameters(): p.requires_grad = True
    for p in bert.pooler.parameters(): p.requires_grad = True
    model = PunctuationCapitalizationRNNBidirectional(
        bert_model=bert,
        hidden_dim=256,
        num_punct_classes=len(PUNCT_TAGS),
        num_cap_classes=len(CAP_TAGS),
        dropout=0.3
    ).to(device)
    state_path = hf_hub_download(repo_id=repo_id, filename="model_bidirec_con_mixture_de_data.pt")
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

model_a = load_classic(MODEL_REPOS["RNN clásica (última versión)"])
model_b = load_bidirectional(MODEL_REPOS["RNN bidireccional (última versión)"])
tokenizer = load_tokenizer()

st.title("Reconstrucción de Texto con Puntuación y Capitalización")

st.info(
    """
✍️ *Instrucciones:*  
Ingresá una frase sin puntuación ni mayúsculas, por ejemplo:  
`hola que lindo dia que hace no te parece`
"""
)

sentence = st.text_input("Ingresá texto:")
if sentence:
    text = sentence.lower()
    result_a = predict_and_reconstruct(model_a, text, tokenizer)
    result_b = predict_and_reconstruct(model_b, text, tokenizer)
    prompt = (
        "Por favor, corrige únicamente la puntuación y la capitalización del siguiente texto, "
        "cumpliendo estas reglas:\n"
        "1. No modifiques el orden de las palabras.\n"
        "2. Solo agrega los signos: ¿, ?, coma (,) y punto (.).\n"
        "3. No introduzcas otros símbolos (¡, !, puntos suspensivos, etc.).\n"
        "4. No añadas acentos que no estuvieran en el texto original.\n"
        "5. No corrijas errores ortográficos ni agregues o elimines palabras.\n\n"
        "Devuelve únicamente el texto transformado, sin explicaciones ni comentarios.\n\n"
        "TEXTO:\n"
        f"{text}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    result_gemini = response.text if (response and response.text) else None
    table_md = "| Modelo | Resultado |\n|---|---|\n"
    table_md += f"| RNN clásica (última versión) | {result_a} |\n"
    table_md += f"| RNN bidireccional (última versión) | {result_b} |\n"
    if result_gemini:
        table_md += f"| Gemini 2.5 Flash | {result_gemini} |\n"
    st.markdown(table_md)

st.markdown("---")
st.markdown(
    "<footer style='text-align:center; color:gray; padding-top:1rem;'>"
    "Esta página es una prueba para un trabajo educativo."  
    "</footer>",
    unsafe_allow_html=True
)
