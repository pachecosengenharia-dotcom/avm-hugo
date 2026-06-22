import streamlit as st
import pandas as pd
import numpy as np
import io
import textwrap
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- Funções Auxiliares ---
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    y = height - 90
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Equação do Modelo:")
    c.setFont("Courier", 8)
    y -= 40
    for linha in textwrap.wrap(eq_str, width=100):
        c.drawString(50, y, linha)
        y -= 12
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Resultados:")
    c.setFont("Helvetica", 10)
    y -= 30
    c.drawString(50, y, f"V.U. Estimado: R$ {d['vu']:,.2f} | Intervalo: R$ {min_v:,.2f} a R$ {max_v:,.2f}")
    y -= 20
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y - 250, width=400, height=200)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- Interface ---
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
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns, key="t")
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target], key="f")
    
    info = {"Endereço": st.sidebar.text_input("Endereço"), "Bairro": st.sidebar.text_input("Bairro")}

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)

            inputs = {}
            for f in features:
                # Limites dinâmicos extraídos da planilha
                min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                
                val = st.sidebar.number_input(f"{f} (Limites: {min_f:.2f} - {max_f:.2f})", value=float(df_c[f].median()), key=f"in_{f}")
                inputs[f] = val
                if val < min_f or val > max_f:
