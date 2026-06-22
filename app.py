import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n, features, fig):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    c.drawString(50, 780, f"Qtd. de Dados: {n}")
    c.drawString(50, 765, f"Equação: {eq_str}")
    c.drawString(50, 750, f"Variáveis: {', '.join(features)}")
    c.drawString(50, 720, f"V.U. Médio: R$ {vu:,.2f} | Mín: R$ {min_v:,.2f} | Máx: R$ {max_v:,.2f}")
    c.drawString(50, 705, f"VALOR TOTAL: R$ {total:,.2f}")
    c.drawString(50, 690, f"Fundamentação: {fund} | Precisão: {prec}")
    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    try:
        df = pd.read_csv(arquivo, encoding='utf-8', sep=None, engine='python')
    except:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    
    df.columns = [normalizar_texto(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            inputs = {}
            for f in features:
                # Trava técnica: 50 a 1500 para qualquer variável com "Setor"
                if "setor" in normalizar_texto(f).lower():
                    min_f, max_f = 50.0,
