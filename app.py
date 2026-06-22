import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def remover_acentos(t):
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

def gerar_laudo_final(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico de Avaliacao (NBR 14653)")
    
    y = 790
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Dados do Imovel:"); y -= 15
    c.setFont("Helvetica", 10)
    for k, v in info.items():
        c.drawString(60, y, remover_acentos(f"{k}: {v}")); y -= 15
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-10, "Equacao do Modelo:"); y -= 25
    c.setFont("Helvetica", 8)
    c.drawString(60, y, remover_acentos(eq[:95]))
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-20, "Variaveis e Parametros:"); y -= 35
    c.setFont("Helvetica", 9)
    for f in features:
        c.drawString(60, y, remover_acentos(f"- {f}: {inputs.get(f, 0):.2f}"))
        y -= 12
        
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-10, "Resultados da Avaliacao:"); y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(60, y, remover_acentos(f"V.U. Medio: R$ {vu:,.2f} | Min: R$ {min_v:,.2f} | Max: R$ {max_v:,.2f}"))
    c.drawString(60, y-15, remover_acentos(f"VALOR TOTAL: R$ {total:,.2f} | Dados: {n}"))
    c.drawString(60, y-30, remover_acentos(f"Fundamentacao: {fund} | Precisao: {prec}"))
    
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliacoes")

st.sidebar.header("Identificacao")
info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [remover_acentos(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Inputs com limites dinâmicos de todas as variáveis
        inputs = {}
        for f in features:
            if "setor" in remover_acentos(f).lower():
                min_f, max_f = 50.0, 1500.0
            else:
                min_f, max_f = float(df[f].min()), float(df[f].max())
            inputs[f] = st.sidebar.number_input(f"{f} (Lim: {min_f:.0f}-{max_f:.0f})", value=float(df[f].median()))

        if st.sidebar.button("Calcular e Gerar Laudo"):
            # Regressão
            modelo = LinearRegression().fit(df[features], df[target])
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            preds = modelo.predict(df[features])
            
            # Cálculos NBR
            residuos = df[target] - preds
            std = np.std(residuos)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            total = vu * inputs[features[0]]
            n, k = len(df), len(features)
            fund, prec = ("Grau III" if n >= 3*k else "Grau I"), ("Grau III" if (max_v-min_v)/(2*vu) <= 0.2 else "Grau I")
            eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

            # Exibição
            st.latex(eq)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("V.U. Min", f"R$ {min_v:,.2f}"); c2.metric("V.U. Med", f"R$ {vu:,.2f}")
            c3.metric("V.U. Max", f"R$ {max_v:,.2f}"); c4.metric("Dados", n)
            st.markdown(f"### Valor Total: R$ {total:,.2f}")
            st.write(f"**Fundamentacao:** {fund} | **Precisao:** {prec}")

            # Gráficos
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(df[target], preds); ax[0].set_title("Aderencia")
            ax[1].scatter(preds, residuos); ax[1].axhline(0, color='red'); ax[1].set_title("Residuos")
            st.pyplot(fig)
            plt.close(fig)

            # PDF
            pdf = gerar_laudo_final(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs)
            st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
