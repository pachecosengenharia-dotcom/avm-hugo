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

# Função de geração do PDF
def gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    c.drawString(50, 780, f"Qtd. de Dados: {n}")
    c.drawString(50, 765, f"Equação: {eq_str[:80]}")
    c.drawString(50, 745, f"V.U. Médio: R$ {vu:,.2f} | Mín: R$ {min_v:,.2f} | Máx: R$ {max_v:,.2f}")
    c.drawString(50, 730, f"VALOR TOTAL: R$ {total:,.2f}")
    c.drawString(50, 715, f"Fundamentação: {fund} | Precisão: {prec}")
    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    try:
        df = pd.read_csv(arquivo, encoding='utf-8', sep=None, engine='python')
    except:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    
    df.columns = [normalizar_texto(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            inputs = {}
            for f in features:
                # Logica de limite forçada para Setor
                if "setor" in normalizar_texto(f).lower():
                    min_f, max_f = 50.0, 1500.0
                else:
                    min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                
                inputs[f] = st.sidebar.number_input(f"{f} (Lim: {min_f:.1f}-{max_f:.1f})", value=float(df_c[f].median()))
            
            if st.sidebar.button("Calcular Precificação"):
                modelo = LinearRegression().fit(df_c[features], df_c[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                
                # NBR 14653: Cálculo de precisão pelo intervalo de confiança
                residuos = df_c[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                
                n, k = len(df_c), len(features)
                # Grau de fundamentação por amostra
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau II" if amp <= 0.3 else "Grau I"
                
                # Construção da equação com formatação segura
                coefs = [f"({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]
                eq_str = f"{target} = {modelo.intercept_:.2f} + " + " + ".join(coefs)

                # Renderização única (evita erros de API)
                st.latex(eq_str)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                c4.metric("Qtd Dados", n)
                
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds)
                ax1.set_title("Aderência")
                ax2.scatter(preds, residuos)
                ax2.axhline(0, color='red')
                ax2.set_title("Resíduos")
                st.pyplot(fig)
                
                pdf = gerar_laudo_pdf(vu, min_v, max_v, fund, prec, eq_str, total, n, features)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
