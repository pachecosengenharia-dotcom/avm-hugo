import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def limpar_texto(t):
    # Remove acentos e retorna string pura para evitar erros no PDF
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

def gerar_laudo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs, fig):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico de Avaliacao (NBR 14653)")
    
    y = 790
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Dados do Imovel:"); y -= 15
    c.setFont("Helvetica", 9)
    for k, v in info.items():
        c.drawString(60, y, limpar_texto(f"{k}: {v}")); y -= 15
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Equacao do Modelo:"); y -= 25
    c.setFont("Helvetica", 8)
    c.drawString(60, y, limpar_texto(eq[:90]))
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-25, "Variaveis e Limites de Extrapolacao:"); y -= 40
    c.setFont("Helvetica", 9)
    for f in features:
        c.drawString(60, y, limpar_texto(f"- {f}: Valor {inputs[f]:.2f}"))
        y -= 12
        
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Resultados:"); y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(60, y, limpar_texto(f"V.U. Medio: R$ {vu:,.2f} | Min: R$ {min_v:,.2f} | Max: R$ {max_v:,.2f}"))
    c.drawString(60, y-15, limpar_texto(f"VALOR TOTAL: R$ {total:,.2f} | Dados: {n} | {fund} | {prec}"))
    
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    c.drawImage(ImageReader(img_data), 50, y-220, width=400, height=180)
    
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliacoes")

st.sidebar.header("Identificacao")
info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [limpar_texto(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        inputs = {}
        for f in features:
            f_limpo = limpar_texto(f)
            # Regra de limites
            if "setor" in f_limpo.lower():
                min_v, max_v = 50.0, 1500.0
            else:
                min_v, max_v = float(df[f].min()), float(df[f].max())
            
            val = st.sidebar.number_input(f"{f_limpo} (Lim: {min_v:.1f}-{max_v:.1f})", value=float(df[f].median()))
            inputs[f] = val
            # Alerta de Extrapolação
            if val < min_v or val > max_v:
                st.sidebar.error(f"Extrapolacao em {f_limpo}!")

        if st.sidebar.button("Calcular e Gerar Laudo"):
            modelo = LinearRegression().fit(df[features],
