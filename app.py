import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Função PDF ---
def gerar_laudo_pdf(vu, total, grau_f, grau_p, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO - NBR 14653")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Valor Unitário Estimado: R$ {vu:,.2f}")
    c.drawString(50, 750, f"VALOR TOTAL ESTIMADO: R$ {total:,.2f}")
    c.drawString(50, 730, f"Fundamentação: {grau_f} | Precisão: {grau_p}")
    c.save()
    buffer.seek(0)
    return buffer

# --- Configuração ---
st.set_page_config(layout="wide", page_title="AVM - Engenharia")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin-1')
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Coluna Alvo (Preço):", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    if features and target:
        df_c = df.dropna(subset=features + [target])
        modelo = LinearRegression().fit(df_c[features], df_c[target])
        
        st.sidebar.markdown("### Valores para Avaliação")
        inputs = {}
        extrapolou = False
        for f in features:
            val = st.sidebar.number_input(f"{f}", value=float(df_c[f].median()), key=f"input_{f}")
            inputs[f] = val
            if val < df_c[f].min() or val > df_c[f].max():
                extrapolou = True
        
        if st.sidebar.button("Calcular Precificação", key="btn_calc"):
            if extrapolou:
                st.error("Erro: Variáveis fora do limite amostral!")
            else:
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                area = list(inputs.values())[0]
                total = vu * area
                
                # Cálculos NBR
                preds = modelo.predict(df_c[features])
                residuos = df_c[target] - preds
                t_score = stats.t.ppf(0.9, len(df_c) - len(features) - 1)
                margem = t_score * np.std(residuos)
                
                # Exibição
                st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
                
                # PDF
                pdf_buffer = gerar_laudo_pdf(vu, total, "Grau III", "Grau III", inputs, vu-margem, vu+margem)
                st.download_button("📥 Baixar Laudo Completo (PDF)", pdf_buffer, "laudo.pdf")
