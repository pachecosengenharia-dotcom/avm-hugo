import streamlit as st
import pandas as pd
import numpy as np
import io
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def limpar(t):
    return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])

def gerar_laudo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs, fig):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico de Avaliacao (NBR 14653)")
    
    y = 790
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Dados do Imovel:"); y -= 15
    c.setFont("Helvetica", 9)
    for k, v in info.items():
        c.drawString(60, y, limpar(f"{k}: {v}")); y -= 12
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Equacao do Modelo:"); y -= 25
    c.setFont("Helvetica", 8)
    c.drawString(60, y, limpar(eq[:90]))
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-20, "Variaveis Utilizadas:"); y -= 35
    c.setFont("Helvetica", 9)
    for f in features:
        c.drawString(60, y, limpar(f"- {f}: {inputs.get(f, 0):.2f}"))
        y -= 12
        
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y-10, "Resultados:"); y -= 25
    c.setFont("Helvetica", 9)
    c.drawString(60, y, limpar(f"V.U. Medio: R$ {vu:,.2f} | Min: R$ {min_v:,.2f} | Max: R$ {max_v:,.2f}"))
    c.drawString(60, y-12, limpar(f"VALOR TOTAL: R$ {total:,.2f} | Dados: {n} | {fund} | {prec}"))
    
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    c.drawImage(ImageReader(img_data), 50, y-180, width=350, height=130)
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliacoes")

# Identificação
st.sidebar.header("Identificacao")
info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Complemento", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding_errors='replace')
    df.columns = [limpar(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variaveis:", options=[c for c in df.columns if c != target])

    if features and target:
        df = df.dropna(subset=features + [target])
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        inputs = {f: st.sidebar.number_input(f"{limpar(f)} (Lim: 50-1500 se Setor)" if "setor" in limpar(f).lower() else limpar(f), value=float(df[f].median())) for f in features}

        # Botão de Calcular
        if st.sidebar.button("Calcular e Guardar Resultados"):
            modelo = LinearRegression().fit(df[features], df[target])
            st.session_state.calc_data = {
                "vu": modelo.predict(np.array([list(inputs.values())]))[0],
                "eq": f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]),
                "inputs": inputs,
                "n": len(df), "features": features
            }
            # Cálculos NBR
            preds = modelo.predict(df[features])
            residuos = df[target] - preds
            std = np.std(residuos)
            st.session_state.calc_data.update({
                "min_v": st.session_state.calc_data["vu"] - (1.96 * std),
                "max_v": st.session_state.calc_data["vu"] + (1.96 * std),
                "total": st.session_state.calc_data["vu"] * inputs[features[0]],
                "fund": "Grau III" if len(df) >= 3*len(features) else "Grau I",
                "prec": "Grau III" if ((st.session_state.calc_data["vu"] + (1.96*std)) - (st.session_state.calc_data["vu"] - (1.96*std)))/(2*st.session_state.calc_data["vu"]) <= 0.2 else "Grau I"
            })

        # Exibição (só ocorre se existirem dados guardados)
        if "calc_data" in st.session_state:
            d = st.session_state.calc_data
            st.latex(d["eq"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("V.U. Min", f"R$ {d['min_v']:,.2f}"); c2.metric("V.U. Med", f"R$ {d['vu']:,.2f}")
            c3.metric("V.U. Max", f"R$ {d['max_v']:,.2f}"); c4.metric
