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

def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    # Tratamento robusto de codificação
    try:
        df = pd.read_csv(arquivo, encoding='utf-8', sep=None, engine='python')
    except:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    
    df.columns = [normalizar_texto(col).strip() for col in df.columns]
    
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns, key="t_alvo")
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target], key="f_exp")
    
    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            # Inputs dinâmicos
            inputs = {}
            for f in features:
                min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                inputs[f] = st.sidebar.number_input(f"{f} (Lim: {min_f:.1f}-{max_f:.1f})", value=float(df_c[f].median()), key=f"in_{f}")

            if st.sidebar.button("Calcular Precificação", key="btn_calc"):
                modelo = LinearRegression().fit(df_c[features], df_c[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                residuos = df_c[target] - preds
                
                # Cálculos fundamentais
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                
                # NBR 14653
                n, k = len(df_c), len(features)
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau II" if amp <= 0.3 else "Grau I"

                # Exibição
                st.latex(f"{target} = {modelo.intercept_:.2f} + " + " + ".join([f"{c:.2f}*{n}" for n, c in zip(features, modelo.coef_)]))
                
                cols = st.columns(4)
                cols[0].metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                cols[1].metric("V.U. Médio", f"R$ {vu:,.2f}")
                cols[2].metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                cols[3].metric("Dados", n)
                
                st.markdown(f"### Valor Total: R$ {total:,.2f}")
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                
                # Gráficos (sem usar ponto e vírgula)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds)
                ax1.set_title("Aderência")
                ax2.scatter(preds, residuos)
                ax2.axhline(0, color='red')
                ax2.set_title("Resíduos")
                st.pyplot(fig)
                # BOTÃO PARA BAIXAR O PDF
            pdf = gerar_laudo_pdf({'vu': vu, 'total': total}, fig, eq_str, info, graus, inputs, min_v, max_v)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
