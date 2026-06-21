import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import unicodedata
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def normalizar(texto):
    return unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def gerar_laudo_pdf(d, fig, eq_str, inputs, info_imovel):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico (NBR 14653)")
    c.setFont("Helvetica", 10)
    y = 790
    for k, v in info_imovel.items():
        c.drawString(50, y, f"{k}: {v}"); y -= 15
    y -= 10
    c.drawString(50, y, "Equacao do Modelo:"); y -= 15
    for i in range(0, len(eq_str), 80):
        c.drawString(50, y, eq_str[i:i+80]); y -= 12
    y -= 10
    c.drawString(50, y, f"RESULTADOS: Min: R$ {d['min']:,.2f} | Medio: R$ {d['vu']:,.2f} | Max: R$ {d['max']:,.2f}")
    c.drawString(50, y-15, f"VALOR TOTAL ESTIMADO: R$ {d['total']:,.2f}")
    if fig is not None:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 50, y-220, width=400, height=180)
    c.save(); buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])
fig = None

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar(c) for c in df.columns]
    
    # Identificação Automática (Lógica de Engenharia)
    target = next((c for c in df.columns if any(x in c for x in ['valor', 'preco', 'unitario'])), None)
    col_area = next((c for c in df.columns if 'area' in c), None)
    
    # Variáveis Explicativas: todas as numéricas, menos o alvo e a área
    features = st.sidebar.multiselect("Variaveis Explicativas:", [c for c in df.columns if c not in [target, col_area]])
    
    st.sidebar.header("📝 Dados do Imovel")
    info_imovel = {k: st.sidebar.text_input(k) for k in ["Endereco", "Bairro", "Informante"]}
    
    if features and target and col_area:
        df_c = df.copy()
        for col in features + [target, col_area]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
