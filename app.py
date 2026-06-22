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
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Sidebar
st.sidebar.header("Dados do Imóvel")
info = {k: st.sidebar.text_input(k) for k in ["Endereço", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])

if arquivo:
    try:
        df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
        df.columns = [normalizar(col).strip() for col in df.columns]
        target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
        features = st.sidebar.multiselect("Variáveis:", options=[c for c in df.columns if c != target])

        if features and target:
            df = df.dropna(subset=features + [target])
            for col in features + [target]:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            inputs = {}
            for f in features:
                if "setor" in normalizar(f).lower():
                    min_f, max_f = 50.0, 1500.0
                else:
                    min_f, max_f = float(df[f].min()), float(df[f].max())
                inputs[f] = st.sidebar.number_input(f"{f} ({min_f:.1f}-{max_f:.1f})", value=float(df[f].median()))

            if st.sidebar.button("Calcular Precificação"):
                modelo = LinearRegression().fit(df[features], df[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df[features])
                
                # Cálculos NBR 14653
                residuos = df[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                n, k = len(df), len(features)
                fund = "Grau III" if n >= 3*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau I"
                eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

                # Exibição completa
                st.latex(eq)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                c4.metric("Qtd Dados", n)
                st.markdown(f"### Valor Total: R$ {total:,.2f}")
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, residuos); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
                st.pyplot(fig)

                # PDF com todos os dados
                buf = io.BytesIO()
                c = canvas.Canvas(buf, pagesize=A4)
                c.drawString(50, 800, f"Laudo: V.U. R${vu:,.2f} | Total: R${total:,.2f}")
                c.drawString(50, 785, f"Dados: {n} | {fund} | {prec}")
                c.save()
                st.download_button("📥 Baixar Laudo", buf.getvalue(), "laudo.pdf")
    except Exception as e:
        st.error(f"Erro: {e}")
