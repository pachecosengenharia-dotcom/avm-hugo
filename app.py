import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Função de geração de PDF corrigida
def gerar_laudo_pdf(d, fig, eq_str, info, graus):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    
    c.setFont("Helvetica", 10)
    y = 770
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15
    
    y -= 10
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    y -= 20
    c.drawString(50, y, f"Equação: {eq_str}")
    y -= 20
    c.drawString(50, y, f"V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    
    # Inserir gráfico (salvo em buffer)
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(img_buf, 50, 400, width=400, height=200)
    
    c.save()
    buffer.seek(0)
    return buffer

# Interface Lateral
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])
info = {
    "Endereço": st.sidebar.text_input("Endereço"),
    "Bairro": st.sidebar.text_input("Bairro")
}

if arquivo is not None:
    df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])

    if features and target:
        df_c = df.dropna(subset=features + [target])
        modelo = LinearRegression().fit(df_c[features], df_c[target])
        eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
        st.latex(eq_str)

        # Inputs de predição
        inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}

        if st.sidebar.button("Calcular"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            total = vu * inputs.get(features[0], 1)
            graus = ("Grau III", "Grau III") # Simplificado para exemplo

            # Gráfico
            fig, ax = plt.subplots()
            ax.scatter(df_c[target], modelo.predict(df_c[features]))
            st.pyplot(fig)

            # Download
            pdf_data = gerar_laudo_pdf({'vu': vu, 'total': total}, fig, eq_str, info, graus)
            st.download_button("📥 Baixar Laudo", pdf_data, "laudo.pdf")
