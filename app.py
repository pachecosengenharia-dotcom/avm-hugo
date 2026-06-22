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

def gerar_laudo_completo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    y = 770
    for k, v in info.items():
        c.drawString(60, y, f"{k}: {v}"); y -= 15
    c.drawString(60, y-10, f"Equação: {eq[:80]}")
    c.drawString(60, y-25, f"V.U. Médio: R$ {vu:,.2f} | Total: R$ {total:,.2f}")
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Sidebar
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
            
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df[f].median())) for f in features}

            if st.sidebar.button("Calcular Precificação"):
                modelo = LinearRegression().fit(df[features], df[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df[features])
                
                # Cálculos
                residuos = df[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]]
                n, k = len(df), len(features)
                fund, prec = ("Grau III" if n >= 3*k else "Grau I"), ("Grau III" if (max_v - min_v)/(2*vu) <= 0.2 else "Grau I")
                eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

                # Exibição
                st.latex(eq)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                c4.metric("Qtd Dados", n)
                
                # Geração do Gráfico FORÇADA
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].scatter(df[target], preds); ax[0].set_title("Aderência")
                ax[1].scatter(preds, residuos); ax[1].axhline(0, color='red'); ax[1].set_title("Resíduos")
                st.pyplot(fig)
                plt.close(fig) # Fecha o gráfico para evitar bugs
                
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                pdf = gerar_laudo_completo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
    except Exception as e:
        st.error(f"Erro: {e}")
