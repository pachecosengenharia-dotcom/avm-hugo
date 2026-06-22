import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def remover_acentos(t):
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

def gerar_laudo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs, fig):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico de Avaliacao (NBR 14653)")
    
    # Dados Imovel
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, 790, "Dados do Imovel:"); y = 775
    c.setFont("Helvetica", 9)
    for k, v in info.items():
        c.drawString(60, y, remover_acentos(f"{k}: {v}")); y -= 12
    
    # Equação
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Equacao do Modelo:"); y -= 25
    c.setFont("Helvetica", 8)
    c.drawString(60, y, remover_acentos(eq[:90]))
    
    # Variaveis
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-20, "Variaveis Utilizadas:"); y -= 35
    c.setFont("Helvetica", 9)
    for f in features:
        val = inputs.get(f, 0) if inputs is not None else 0
        c.drawString(60, y, remover_acentos(f"- {f}: {val:.2f}"))
        y -= 12
        
    # Resultados
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Resultados:"); y -= 25
    c.setFont("Helvetica", 9)
    c.drawString(60, y, remover_acentos(f"V.U. Medio: R$ {vu:,.2f} | Min: R$ {min_v:,.2f} | Max: R$ {max_v:,.2f}"))
    c.drawString(60, y-12, remover_acentos(f"VALOR TOTAL: R$ {total:,.2f} | Dados: {n} | {fund} | {prec}"))
    
    # Graficos
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    c.drawImage(ImageReader(img_data), 50, y-180, width=350, height=130)
    
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliacoes")

# Identificação
info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [remover_acentos(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        inputs = {}
        for f in features:
            f_clean = remover_acentos(f)
            min_v, max_v = (50.0, 1500.0) if "setor" in f_clean.lower() else (float(df[f].min()), float(df[f].max()))
            inputs[f] = st.sidebar.number_input(f"{f_clean} (Lim: {min_v:.0f}-{max_v:.0f})", value=float(df[f].median()))
            if inputs[f] < min_v or inputs[f] > max_v: st.sidebar.warning(f"Extrapolacao em {f_clean}!")

        if st.sidebar.button("Calcular e Gerar Laudo"):
            modelo = LinearRegression().fit(df[features], df[target])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            preds = modelo.predict(df[features])
            
            # Estatistica
            residuos = df[target] - preds
            std = np.std(residuos)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            total = vu * inputs[features[0]]
            n, k = len(df), len(features)
            fund, prec = ("Grau III" if n >= 3*k else "Grau I"), ("Grau III" if (max_v-min_v)/(2*vu) <= 0.2 else "Grau I")
            eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

            # Exibicao
            st.latex(eq)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("V.U. Min", f"R$ {min_v:,.2f}"); c2.metric("V.U. Med", f"R$ {vu:,.2f}")
            c3.metric("V.U. Max", f"R$ {max_v:,.2f}"); c4.metric("Dados", n)
            st.markdown(f"### Valor Total: R$ {total:,.2f}")
            st.write(f"**Fundamentacao:** {fund} | **Precisao:** {prec}")

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(df[target], preds); ax[0].set_title("Aderencia")
            ax[1].scatter(preds, residuos); ax[1].axhline(0, color='red'); ax[1].set_title("Residuos")
            st.pyplot(fig)
            
            pdf = gerar_pdf(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs, fig)
            st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
            plt.close(fig)
