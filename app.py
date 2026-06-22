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

# --- Funções ---
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"V.U. Estimado: R$ {d['vu']:,.2f}")
    c.drawString(50, 750, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
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
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns, key="t_alvo")
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target], key="f_exp")

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
                # O limite é lido diretamente do CSV, sem valores fixos
                min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                val = st.sidebar.number_input(f"{f} (Lim: {min_f:.1f} - {max_f:.1f})", value=float(df_c[f].median()), key=f"in_{f}")
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.warning(f"⚠️ Extrapolação em {f}!")

            if st.sidebar.button("Calcular Precificação", key="btn_calc"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                
                residuos = df_c[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]]
                
                # NBR 14653
                n, k = len(df_c), len(features)
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amplitude
