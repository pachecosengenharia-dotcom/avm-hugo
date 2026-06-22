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

def limpar(t):
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

def gerar_laudo_pdf(res, fig, eq_str, info, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Tecnico de Avaliacao (NBR 14653)")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, 770, "Dados do Imovel:"); y = 755
    c.setFont("Helvetica", 9)
    for k, v in info.items():
        c.drawString(60, y, limpar(f"{k}: {v}")); y -= 12
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Equacao do Modelo:"); y -= 25
    c.setFont("Helvetica", 8)
    c.drawString(60, y, limpar(eq_str[:90]))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-20, "Variaveis Utilizadas:"); y -= 35
    c.setFont("Helvetica", 9)
    for f in inputs:
        c.drawString(60, y, limpar(f"- {f}: {inputs[f]:.2f}"))
        y -= 12
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Resultados:"); y -= 25
    c.setFont("Helvetica", 9)
    c.drawString(60, y, limpar(f"V.U. Medio: R$ {res['vu']:,.2f} | Min: R$ {res['min_v']:,.2f} | Max: R$ {res['max_v']:,.2f}"))
    c.drawString(60, y-12, limpar(f"VALOR TOTAL: R$ {res['total']:,.2f} | {res['fund']} | {res['prec']}"))
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y-180, width=350, height=130)
    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliacoes")

info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [limpar(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        inputs = {}
        for f in features:
            f_clean = limpar(f)
            min_v, max_v = (50.0, 1500.0) if "setor" in f_clean.lower() else (float(df[f].min()), float(df[f].max()))
            inputs[f] = st.sidebar.number_input(f"{f_clean} (Lim: {min_v:.0f}-{max_v:.0f})", value=float(df[f].median()))

        if st.sidebar.button("Calcular Precificacao"):
            modelo = LinearRegression().fit(df[features], df[target])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            preds = modelo.predict(df[features])
            residuos = df[target] - preds
            std = np.std(residuos)
            
            res = {
                "vu": vu, "min_v": vu - 1.96*std, "max_v": vu + 1.96*std,
                "total": vu * inputs[features[0]], "n": len(df),
                "fund": "Grau III" if len(df) >= 3*len(features) else "Grau I",
                "prec": "Grau III" if ((vu+1.96*std)-(vu-1.96*std))/(2*vu) <= 0.2 else "Grau I"
            }
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            
            st.latex(eq_str)
            st.markdown(f"### Valor Total: R$ {res['total']:,.2f}")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(df[target], preds); ax[1].scatter(preds, residuos); st.pyplot(fig)
            
            pdf = gerar_laudo_pdf(res, fig, eq_str, info, inputs)
            st.download_button("📥 Baixar Laudo Completo", pdf, "laudo.pdf")
            plt.close(fig)
