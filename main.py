import os
import streamlit as st
import torch
from transformers import BertModel, BertTokenizerFast
from huggingface_hub import hf_hub_download
from model import PunctuationCapitalizationRNN, PUNCT_TAGS, CAP_TAGS
from inference_utils import predict_and_reconstruct

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
sentence = st.text_input("Ingresá texto:")

if sentence:
    result = predict_and_reconstruct(model, sentence, tokenizer)
    st.write("**Resultado:**", result)
    st.download_button(
        label="Descargar resultado",
        data=result.encode("utf-8"),
        file_name="prediction.txt",
        mime="text/plain"
    )
