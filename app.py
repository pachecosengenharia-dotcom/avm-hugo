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

def gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n, info):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    y = 770
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15
    c.drawString(50, y-10, f"Qtd Dados: {n} | Fundamentação: {fund} | Precisão: {prec}")
    c.drawString(50, y-25, f"V.U. Médio: R$ {vu:,.2f} | Total: R$ {total:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

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
    try:
        # Tenta ler o arquivo de forma robusta
        df = pd.read_csv(arquivo, sep=None, engine='python', on_bad_lines='skip')
        df.columns = [normalizar_texto(col).strip() for col in df.columns]
        
        target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
        features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

        if features and target:
            df_c = df.dropna(subset=features + [target])
            inputs = {}
            for f in features:
                if "setor" in normalizar_texto(f).lower():
                    min_f, max_f = 50.0, 1500.0
                else:
                    min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                inputs[f] = st.sidebar.number_input(f"{f} (Lim: {min_f:.1f}-{max_f:.1f})", value=float(df_c[f].median()))

            if st.sidebar.button("Calcular Precificação"):
                modelo = LinearRegression().fit(df_c[features], df_c[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                
                residuos = df_c[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]]
                
                n, k = len(df_c), len(features)
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau II" if amp <= 0.3 else "Grau I"
                eq_str = f"{target} = {modelo.intercept_:.2f}"

                st.latex(eq_str)
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                
                pdf = gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n, info)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
