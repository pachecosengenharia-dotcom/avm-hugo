import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Remove acentos para evitar erros de renderização no PDF
def remover_acentos(texto):
    if not isinstance(texto, str): texto = str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def gerar_laudo_completo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 820, "Laudo Tecnico de Avaliacao (NBR 14653)")
    
    # Dados Imovel
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 790, "Dados do Imovel:")
    c.setFont("Helvetica", 10)
    y = 770
    for k, v in info.items():
        c.drawString(60, y, remover_acentos(f"{k}: {v}")); y -= 15
    
    # Dados Tecnicos
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-10, "Equacao do Modelo:")
    c.setFont("Helvetica", 9)
    c.drawString(60, y-25, remover_acentos(eq[:90]))
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-50, "Variaveis e Parametros:")
    c.setFont("Helvetica", 10)
    y_var = y-65
    for f in features:
        c.drawString(60, y_var, remover_acentos(f"- {f}: {inputs[f]:.2f}"))
        y_var -= 13
        
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_var-10, "Resultados da Avaliacao:")
    c.setFont("Helvetica", 10)
    c.drawString(60, y_var-25, remover_acentos(f"V.U. Medio: R$ {vu:,.2f} | Min: R$ {min_v:,.2f} | Max: R$ {max_v:,.2f}"))
    c.drawString(60, y_var-37, remover_acentos(f"VALOR TOTAL ESTIMADO: R$ {total:,.2f}"))
    c.drawString(60, y_var-49, remover_acentos(f"Fundamentacao: {fund} | Precisao: {prec} | Total de Dados: {n}"))
    
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
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        inputs = {f: st.sidebar.number_input(f"{f} (Lim: 50-1500 se Setor)" if "setor" in remover_acentos(f).lower() else f"{f}", value=float(df[f].median())) for f in features}

        if st.sidebar.button("Calcular Precificacao"):
            modelo = LinearRegression().fit(df[features], df[target])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            preds = modelo.predict(df[features
