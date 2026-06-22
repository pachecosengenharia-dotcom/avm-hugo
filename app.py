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

# Função de geração do PDF atualizada com todos os campos
def gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n, info):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    
    # Adicionando os novos campos de identificação
    c.setFont("Helvetica", 10)
    y = 770
    for key, value in info.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 15
        
    c.drawString(50, y-10, f"Qtd. de Dados: {n}")
    c.drawString(50, y-25, f"Equação: {eq_str[:80]}")
    c.drawString(50, y-40, f"V.U. Médio: R$ {vu:,.2f} | Mín: R$ {min_v:,.2f} | Máx: R$ {max_v:,.2f}")
    c.drawString(50, y-55, f"VALOR TOTAL: R$ {total:,.2f}")
    c.drawString(50, y-70, f"Fundamentação: {fund} | Precisão: {prec}")
    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações")

# Captura de dados do imóvel na barra lateral
st.sidebar.header("Identificação do Imóvel")
info = {
    "Endereço": st.sidebar.text_input("Endereço"),
    "Complemento": st.sidebar.text_input("Complemento"),
    "Bairro": st.sidebar.text_input("Bairro"),
    "Informante": st.sidebar.text_input("Informante"),
    "Telefone": st.sidebar.text_input("Telefone")
}

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    # ... (código de leitura do CSV)
    try:
        df = pd.read_csv(arquivo, encoding='utf-8', sep=None, engine='python')
    except:
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
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                modelo = LinearRegression().fit(df_c[features], df_c[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                
                # Cálculos NBR 14653
                residuos = df_c[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                n, k = len(df_c), len(features)
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau II" if amp <= 0.3 else "Grau I"
                eq_str = f"{target} = {modelo.intercept_:.2f} + " + " + ".join([f"{c:.2f}*{n}" for n, c in zip(features, modelo.coef_)])

                # Exibição
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                
                # Geração do PDF garantida
                pdf = gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n, info)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
