import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def normalizar(t):
    return unicodedata.normalize('NFKD', str(t))

def gerar_laudo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 820, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    
    # Identificação
    y = 790
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}"); y -= 15
    
    # Dados Técnicos
    c.drawString(50, y-20, f"Equação: {eq[:80]}...")
    c.drawString(50, y-35, f"Total Dados: {n} | V.U. Médio: R$ {vu:,.2f}")
    c.drawString(50, y-50, f"Valor Total: R$ {total:,.2f} | Fundamentação: {fund} | Precisão: {prec}")
    
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

st.sidebar.header("Identificação")
info = {k: st.sidebar.text_input(k) for k in ["Endereço", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [normalizar(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        inputs = {f: st.sidebar.number_input(f"{f} (Lim: 50-1500 se Setor)" if "setor" in normalizar(f).lower() else f"{f}", value=float(df[f].median())) for f in features}

        if st.sidebar.button("Calcular Precificação"):
            # Processamento
            modelo = LinearRegression().fit(df[features], df[target])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            preds = modelo.predict(df[features])
            
            # Estatística
            residuos = df[target] - preds
            std = np.std(residuos)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            total = vu * inputs[features[0]]
            n, k = len(df), len(features)
            fund = "Grau III" if n >= 3*k else "Grau I"
            amp = (max_v - min_v) / (2 * vu)
            prec = "Grau III" if amp <= 0.2 else "Grau I"
            eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

            # Renderização
            st.latex(eq)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("V.U. Min", f"R$ {min_v:,.2f}"); c2.metric("V.U. Med", f"R$ {vu:,.2f}")
            c3.metric("V.U. Max", f"R$ {max_v:,.2f}"); c4.metric("Dados", n)
            st.markdown(f"### Valor Total: R$ {total:,.2f}")
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(df[target], preds); ax[0].set_title("Aderencia")
            ax[1].scatter(preds, residuos); ax[1].axhline(0, color='red'); ax[1].set_title("Residuos")
            st.pyplot(fig)
            plt.close(fig)

            # Botão de Download
            pdf = gerar_laudo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs)
            st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
