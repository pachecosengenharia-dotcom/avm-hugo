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

def remover_acentos(t):
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

def gerar_laudo_completo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs, fig):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico de Avaliacao (NBR 14653)")
    
    # Identificacao
    y = 790
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Dados do Imovel:"); y -= 15
    c.setFont("Helvetica", 9)
    for k, v in info.items():
        c.drawString(60, y, remover_acentos(f"{k}: {v}")); y -= 15
    
    # Equacao e Resultados
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Equacao do Modelo:"); y -= 25
    c.setFont("Helvetica", 8)
    c.drawString(60, y, remover_acentos(eq[:90]))
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-20, "Resultados:"); y -= 35
    c.setFont("Helvetica", 9)
    c.drawString(60, y, remover_acentos(f"VU Medio: R$ {vu:,.2f} | Min: R$ {min_v:,.2f} | Max: R$ {max_v:,.2f}"))
    c.drawString(60, y-15, remover_acentos(f"VALOR TOTAL: R$ {total:,.2f} | Dados: {n} | {fund} | {prec}"))
    
    # Insercao do Grafico no PDF
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    c.drawImage(ImageReader(img_data), 50, y-200, width=400, height=150)
    
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliacoes")

# Sidebar
st.sidebar.header("Identificacao")
info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [remover_acentos(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        inputs = {f: st.sidebar.number_input(f"{remover_acentos(f)} (Lim: 50-1500 se Setor)" if "setor" in remover_acentos(f).lower() else remover_acentos(f), value=float(df[f].median())) for f in features}

        if st.sidebar.button("Calcular e Gerar Laudo"):
            modelo = LinearRegression().fit(df[features], df[target])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            preds = modelo.predict(df[features])
            
            # Estatistica
            residuos = df[target] - preds
            std = np.std(residuos)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            total = vu * inputs[features[0]]
            n, k = len(df), len(features)
            fund, prec = ("Grau III" if n >= 3*k else "Grau I"), ("Grau III" if (max_v-min_v)/(2*vu) <= 0.2
