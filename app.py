import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Funções básicas
def normalizar(t):
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

# Configuração da página
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Sidebar - Identificação
st.sidebar.header("Identificação")
info = {
    "Endereço": st.sidebar.text_input("Endereço"),
    "Complemento": st.sidebar.text_input("Complemento"),
    "Bairro": st.sidebar.text_input("Bairro"),
    "Informante": st.sidebar.text_input("Informante"),
    "Telefone": st.sidebar.text_input("Telefone")
}

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])

if arquivo:
    try:
        # Leitura ultra-robusta
        df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
        df.columns = [normalizar(col).strip() for col in df.columns]
        
        target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
        features = st.sidebar.multiselect("Variáveis:", options=[c for c in df.columns if c != target])

        if features and target:
            df = df.dropna(subset=features + [target])
            for col in features + [target]:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            inputs = {}
            for f in features:
                if "setor" in normalizar(f).lower():
                    min_f, max_f = 50.0, 1500.0
                else:
                    min_f, max_f = float(df[f].min()), float(df[f].max())
                inputs[f] = st.sidebar.number_input(f"{f} ({min_f}-{max_f})", value=float(df[f].median()))

            if st.sidebar.button("Calcular"):
                modelo = LinearRegression().fit(df[features], df[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                
                # NBR 14653 simplificada
                n, k = len(df), len(features)
                fund = "Grau III" if n >= 3*k else "Grau I"
                
                # Exibição
                st.metric("V.U. Estimado", f"R$ {vu:,.2f}")
                st.write(f"Fundamentação: {fund}")
                
                # PDF simplificado
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                c.drawString(50, 800, f"Laudo - V.U.: R$ {vu:,.2f}")
                c.save()
                st.download_button("📥 Baixar PDF", buffer.getvalue(), "laudo.pdf")
                
    except Exception as e:
        st.error(f"Erro no processamento: {e}")
