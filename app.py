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

# Função que gera o PDF com todos os campos da NBR 14653
def gerar_laudo_completo(vu, min_v, max_v, fund, prec, eq, total, n, info, features, inputs):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 820, "Laudo Técnico de Avaliação (NBR 14653)")
    
    # Dados do Imóvel
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 790, "Dados do Imóvel:")
    c.setFont("Helvetica", 10)
    y = 770
    for k, v in info.items():
        c.drawString(60, y, f"{k}: {v}"); y -= 15
    
    # Equação
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-10, "Equação do Modelo:")
    c.setFont("Helvetica", 9)
    # Quebra a equação em linhas para caber no PDF
    c.drawString(60, y-25, eq[:90])
    c.drawString(60, y-37, eq[90:])
    
    # Variáveis utilizadas
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-60, "Variáveis e Parâmetros Utilizados:")
    c.setFont("Helvetica", 10)
    y_var = y-75
    for f in features:
        c.drawString(60, y_var, f"- {f}: {inputs[f]:.2f}")
        y_var -= 13
        
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_var-10, "Resultados da Avaliação:")
    c.setFont("Helvetica", 10)
    c.drawString(60, y_var-25, f"V.U. Médio: R$ {vu:,.2f} | Mín: R$ {min_v:,.2f} | Máx: R$ {max_v:,.2f}")
    c.drawString(60, y_var-37, f"VALOR TOTAL ESTIMADO: R$ {total:,.2f}")
    c.drawString(60, y_var-49, f"Fundamentação: {fund} | Precisão: {prec} | Total de Dados: {n}")
    
    c.save()
    buf.seek(0)
    return buf

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

# Sidebar: Entradas
st.sidebar.header("Identificação")
info = {k: st.sidebar.text_input(k) for k in ["Endereço", "Bairro", "Informante", "Telefone"]}
arquivo = st.sidebar.file_uploader("Base (CSV)", type=["csv", "txt"])

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
                min_f, max_f = (50.0, 1500.0) if "setor" in normalizar(f).lower() else (float(df[f].min()), float(df[f].max()))
                inputs[f] = st.sidebar.number_input(f"{f} ({min_f:.0f}-{max_f:.0f})", value=float(df[f].median()))

            if st.sidebar.button("Calcular Laudo"):
                modelo = LinearRegression().fit(df[features], df[target])
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df[features])
                
                # Cálculos Estatísticos
                residuos = df[target] - preds
                std = np.std(residuos)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]]
                n, k = len(df), len(features)
                
                # Critérios NBR
                fund = "Grau III" if n >= 3*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau I"
                eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

                # Layout de Exibição
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

                pdf = gerar_laudo_completo(vu, min_v, max_v, fund, prec, eq, total, n, info, features)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
    except Exception as e:
        st.error(f"Erro técnico: {e}")
