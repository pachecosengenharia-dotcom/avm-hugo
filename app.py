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

# Função para remover acentos
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

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

    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns, key="target_col")
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target], key="features_col")

    st.sidebar.header("📝 Dados do Imóvel")
    info = {
        "Endereço": st.sidebar.text_input("Endereço", key="info_end"),
        "Bairro": st.sidebar.text_input("Bairro", key="info_bairro"),
        "Informante": st.sidebar.text_input("Informante", key="info_inf")
    }

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)

            inputs = {}
            st.sidebar.markdown("### Valores para Avaliação")
            for f in features:
                min_f, max_f = df_c[f].min(), df_c[f].max()
                val = st.sidebar.number_input(
                    f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", 
                    value=float(df_c[f].median()), 
                    key=f"inp_{f}"
                )
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.warning(f"⚠️ Extrapolação em {f}!")

            if st.sidebar.button("Calcular Precificação", key="btn_calc"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                graus = ("Grau III" if len(df_c) >= 12 else "Grau I", "Grau III" if (max_v-min_v)/(2*vu) < 0.2 else "Grau I")

                cols = st.columns(3)
                cols[0].metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                cols[1].metric("V.U. Médio", f"R$ {vu:,.2f}")
                cols[2].metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
                st.pyplot(fig)
                
                # (Nota: A função gerar_laudo_pdf deve estar definida antes deste bloco)
                st.success("Cálculo realizado com sucesso!")
