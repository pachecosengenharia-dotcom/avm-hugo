import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def limpar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def calcular_graus(n, n_vars, intervalo_relativo):
    fund = "Grau I" if n < 6*n_vars else ("Grau II" if n < 12*n_vars else "Grau III")
    prec = "Grau I" if intervalo_relativo > 0.30 else ("Grau II" if intervalo_relativo > 0.20 else "Grau III")
    return fund, prec

def gerar_laudo_pdf(d, fig, eq_str, info_extra, graus):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 800, "Laudo Tecnico de Avaliacao")
    c.setFont("Helvetica", 10)
    y = 770
    for label, val in info_extra.items():
        c.drawString(50, y, f"{label}: {val}")
        y -= 15
    y -= 10
    c.drawString(50, y, f"Fundamentacao: {graus[0]} | Precisao: {graus[1]}")
    y -= 20
    c.drawString(50, y, f"Equacao: {eq_str[:80]}")
    y -= 20
    c.drawString(50, y, f"V.U. Medio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 300, width=400, height=200)
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python')
    df.columns = [limpar_texto(col.strip()) for col in df.columns]
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variaveis:", [c for c in df.columns if c != target])
    
    info_extra = {
        "Endereco": st.sidebar.text_input("Endereco"),
        "Bairro": st.sidebar.text_input("Bairro")
    }

    if features and target:
        df_c = df.dropna(subset=features+[target])
        modelo = LinearRegression().fit(df_c[features], df_c[target])
        
        eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
        st.latex(eq_str)
        
        st.sidebar.header("⚙️ Parametros")
        inputs = {f: st.sidebar.number_input(f"{f} (Lim: {df_c[f].min():.1f} a {df_c[f].max():.1f})", 
                  value=float(df_c[f].median())) for f in features}
        
        if st.sidebar.button("Calcular Precificacao"):
            preds = modelo.predict(df_c[features])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            std = np.std(df_c[target] - preds)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            graus = calcular_graus(len(df_c), len(features), (max_v - min_v) / (2 * vu))
            
            # Graficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.scatter(df_c[target], preds); ax1.set_title("Aderencia")
            ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Residuos")
            st
