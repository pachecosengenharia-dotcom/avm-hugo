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

# Função para normalizar texto
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# Função para Gerar o PDF
def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    c.drawString(50, 780, f"V.U. Estimado: R$ {d['vu']:,.2f}")
    c.drawString(50, 760, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
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
                min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                val = st.sidebar.number_input(f"{f} (Lim: {min_f:.1f}-{max_f:.1f})", value=float(df_c[f].median()), key=f"in_{f}")
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.warning(f"⚠️ Extrapolação em {f}!")

            if st.sidebar.button("Calcular Precificação", key="btn_calc"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                
                # Cálculos fundamentais definidos aqui
                residuos = df_c[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]]
                
                # NBR 14653 - Definindo as variáveis ANTES de usá-las
                n, k = len(df_c), len(features)
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amplitude = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amplitude <= 0.2 else "Grau II" if amplitude <= 0.3 else "Grau I"
                
                # Exibição
                cols = st.columns(3)
                cols[0].metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                cols[1].metric("V.U. Médio", f"R$ {vu:,.2f}")
                cols[2].metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds)
                ax1.set_title("Aderência")
                ax2.scatter(preds, residuos)
                ax2.axhline(0, color='red')
                ax2.set_title("Resíduos")
                st.pyplot(fig)
                
                # PDF
                pdf = gerar_laudo_pdf({'vu': vu}, fig, eq_str, {"Bairro": "N/A"}, (fund, prec), inputs, min_v, max_v)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
