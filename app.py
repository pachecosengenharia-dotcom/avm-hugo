import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# DEFINIÇÃO DA FUNÇÃO: 7 ARGUMENTOS
def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, limites):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    
    y = 790
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    y -= 20
    c.drawString(50, y, f"Equação: {eq_str[:80]}")
    y -= 20
    c.drawString(50, y, f"V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 300, width=400, height=200)
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
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
        df_c = df.dropna(subset=features + [target])
        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            st.sidebar.header("⚙️ Parâmetros")
            limites = {f: {'min': float(df_c[f].min()), 'max': float(df_c[f].max())} for f in features}
            inputs = {f: st.sidebar.number_input(f"{f} (Lim: {limites[f]['min']:.1f} a {limites[f]['max']:.1f})", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                graus = ("Grau III" if len(df_c) >= 12 else "Grau I", "Grau III" if (max_v-min_v)/(2*vu) < 0.
