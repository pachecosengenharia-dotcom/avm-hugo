import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def limpar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def calcular_graus(n, n_vars, intervalo_relativo):
    fund = "Grau I" if n < 6*n_vars else ("Grau II" if n < 12*n_vars else "Grau III")
    prec = "Grau I" if intervalo_relativo > 0.30 else ("Grau II" if intervalo_relativo > 0.20 else "Grau III")
    return fund, prec

def gerar_laudo_pdf(d, fig, eq_str, inputs, info_extra, variaveis_limites, graus):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Técnico de Avaliação")
    c.setFont("Helvetica", 10)
    y = 790
    for label, val in info_extra.items():
        c.drawString(50, y, f"{label}: {val}")
        y -= 15
    y -= 10
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    y -= 25
    c.drawString(50, y, f"V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [limpar_texto(col.strip()) for col in df.columns]
    
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis:", [c for c in df.columns if c != target])
    
    info_extra = {
        "Endereco": st.sidebar.text_input("Endereco"),
        "Bairro": st.sidebar.text_input("Bairro"),
        "Informante": st.sidebar.text_input("Informante")
    }

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            st.sidebar.header("📊 Parâmetros")
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                preds = modelo.predict(df_c[features])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                
                col_area = next((c for c in features if 'area' in c.lower()), None)
                total = vu * inputs[col_area] if col_area else vu
                graus = calcular_graus(len(df_c), len(features), (max_v - min_v) / (2 * vu))
                
                st.info(f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Min", f"R$ {min_v:,.2f}")
                c2.metric("Med", f"R$ {vu:,.2f}")
                c3.metric("Max", f"R$ {max_v:,.2f}")
                
                st.markdown(f"### Valor Total: R$ {total:,.2f}")
                
                fig, ax = plt.subplots()
                ax.scatter(df_c[target], preds)
                st.pyplot(fig)
