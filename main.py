import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import torch
from transformers import BertModel, BertTokenizerFast
from huggingface_hub import hf_hub_download
from model import PunctuationCapitalizationRNN, PUNCT_TAGS, CAP_TAGS
from utils import predict_and_reconstruct, save_feedback
from google import genai
from google.genai import types

client = genai.Client(
    api_key= st.secrets["GOOGLE_GENAI_API_KEY"] or os.getenv("GOOGLE_GENAI_API_KEY"),
)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Reconstruís BERT y aplicás freeze/fine-tuning parcial
    bert = BertModel.from_pretrained("bert-base-multilingual-cased")
    for p in bert.embeddings.word_embeddings.parameters(): p.requires_grad = False
    for layer in bert.encoder.layer[-2:]:
        for p in layer.parameters(): p.requires_grad = True
    for p in bert.pooler.parameters(): p.requires_grad = True

    # 2) Instanciás tu RNN
    model = PunctuationCapitalizationRNN(
        bert_model=bert,
        hidden_dim=256,
        num_punct_classes=len(PUNCT_TAGS),
        num_cap_classes=len(CAP_TAGS),
        dropout=0.3
    ).to(device)

    # 3) Descargás el state_dict desde HF Hub
    #    repo_id = "fabroo/PunctuationCapitalizationRNN"
    state_path = hf_hub_download(
        repo_id="fabroo/PunctuationCapitalizationRNN",
        filename="modelo_fine_tuned_state_dict.pt"
    )
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4) Modo evaluación
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

model     = load_model()
tokenizer = load_tokenizer()

st.title("Reconstrucción de Texto con Puntuación y Capitalización")

st.info("""
✍️ *Instrucciones:*  
Ingresá una frase sin puntuación ni mayúsculas, por ejemplo:  
`hola que lindo dia que hace no te parece`

El modelo la corregirá automáticamente:  
`Hola, qué lindo día que hace, ¿no te parece?`
""")

st.markdown("""
    <style>
    input[type="text"] {
        text-transform: lowercase;
    }
    </style>
""", unsafe_allow_html=True)

sentence = st.text_input("Ingresá texto:")
sentence = sentence.lower()

if sentence:
    result = predict_and_reconstruct(model, sentence.lower(), tokenizer)
    st.write("**Resultado Modelo:**", result)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Por favor, revisa el siguiente texto y realiza las correcciones necesarias en puntuación y capitalización:\n\n" + result 
        + "\n\nAsegúrate de que el texto esté correctamente puntuado y capitalizado."
        + "\n\n**Nota:** No agregues ni elimines palabras, solo corrige la puntuación y la capitalización."
        + "\n\n SOLO DEVOLVÉ EL TEXTO CORREGIDO, SIN EXPLICACIONES NI COMENTARIOS.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    if response and response.text:
        st.write("**Resultado Gemini 2.5 Flash:**", response.text)


    st.markdown("💡 ¿El resultado no fue correcto? Podés enviarnos tu feedback con el botón de abajo:")

    if st.button("📬 Enviar feedback: el modelo se equivocó"):
        save_feedback(sentence, result)
        st.success("✅ ¡Gracias! Tu feedback fue guardado.")
