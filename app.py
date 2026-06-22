import streamlit as st
import pandas as pd
import numpy as np
import io
import textwrap
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configuração da Página
st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações")

# Função para Gerar o PDF
def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Cabeçalho
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Laudo Técnico de Avaliação (NBR 14653)")
    
    # Informações do Imóvel
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 90, "Dados do Imóvel:")
    c.setFont("Helvetica", 10)
    y = height - 110
    for k, v in info.items():
        # Decodifica para garantir acentos no PDF
        texto = f"{k}: {v}"
        c.drawString(50, y, texto)
        y -= 20
    
    # Equação do Modelo
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Equação do Modelo:")
    y -= 20
    c.setFont("Courier", 8)
    linhas_eq = textwrap.wrap(eq_str, width=100)
    for linha in linhas_eq:
        c.drawString(50, y, linha)
        y -= 12
    
    # Variáveis Utilizadas
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Variáveis e Parâmetros Utilizados:")
    y -= 20
    c.setFont("Helvetica", 10)
    for k, v in inputs.items():
        c.drawString(50, y, f"- {k}: {v:.2f}")
        y -= 15
    
    # Resultados Finais
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Resultados da Avaliação:")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(50, y, f"V.U. Mínimo: R$ {min_v:,.2f} | V.U. Médio: R$ {d['vu']:,.2f} | V.U. Máximo: R$ {max_v:,.2f}")
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"VALOR TOTAL ESTIMADO: R$ {d['total']:,.2f}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    
    # Inserção do Gráfico
    y -= 220
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y, width=400, height=200)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Interface Lateral
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    # Correção de Encoding: forçando utf-8 para dispositivos móveis
    try:
        df = pd.read_csv(arquivo, encoding='utf-8', sep=None, engine='python')
    except:
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
        
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    st.sidebar.header("📝 Dados do Imóvel")
    info = {
        "Endereço": st.sidebar.text_input("Endereço"),
        "Bairro": st.sidebar.text_input("Bairro"),
        "Informante": st.sidebar.text_input("Informante")
    }

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})"
