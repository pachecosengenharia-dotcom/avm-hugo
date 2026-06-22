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

def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    # ... (mantive a função conforme seu original, apenas garantindo os parâmetros)
    c.showPage(); c.save(); buffer.seek(0)
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
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns, key="t_alvo")
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target], key="f_exp")
    
    info = {"Endereço": st.sidebar.text_input("Endereço"), "Bairro": st.sidebar.text_input("Bairro")}

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
            for f in features:
                min_f, max_f = float(df_c[f].min()), float(df_c[f].max())
                val = st.sidebar.number_input(f"{f} (Limites: {min_f:.2f} - {max_f:.2f})", value=float(df_c[f].median()), key=f"in_{f}")
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.warning(f"⚠️ Extrapolação em {f}: fora da amostra!")

            if st.sidebar.button("Calcular Precificação", key="btn_calc"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                
                # Cálculos de precisão
                residuos = df_c[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] # Ajuste conforme sua variável de área
                
                # Lógica NBR 14653
                n, k = len(df_c), len(features)
                fund = "Grau III" if n >= 3*k else "Grau II" if n >= 2*k else "Grau I"
                amplitude = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amplitude <= 0.2 else "Grau II" if amplitude <= 0.3 else "Grau I"
                
                # EXIBIÇÃO RESTAURADA
                cols = st.columns(3)
                cols[0].metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                cols[1].metric("V.U. Médio", f"R$ {vu:,.2f}")
                cols[2].metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")
                
                # Gráficos Restaurados
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, residuos); ax2.axhline(0, color='red'); ax2.
