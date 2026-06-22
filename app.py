import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configuração da Página
st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações")

# Função para Gerar o PDF
def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Cabeçalho
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Laudo Técnico de Avaliação (NBR 14653)")
    
    # Informações do Imóvel
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 90, "Dados do Imóvel:")
    c.setFont("Helvetica", 10)
    y = height - 110
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20
    
    # Modelo Matemático
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Equação do Modelo:")
    c.setFont("Courier", 9)
    y -= 20
    c.drawString(50, y, eq_str)
    
    # Resultados
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Resultados da Avaliação:")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(50, y, f"V.U. Mínimo: R$ {min_v:,.2f} | V.U. Médio: R$ {d['vu']:,.2f} | V.U. Máximo: R$ {max_v:,.2f}")
    y -= 20
    c.drawString(50, y, f"Valor Total Estimado: R$ {d['total']:,.2f}")
    y -= 20
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    
    # Gráficos
    y -= 280
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y, width=480, height=250)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Interface Lateral
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    st.sidebar.header("📝 Dados do Imóvel")
    info = {
        "Endereço": st.sidebar.text_input("Endereço"),
        "Bairro": st.sidebar.text_input("Bairro"),
        "Informante": st.sidebar.text_input("Informante")
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
            
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                graus = ("Grau III" if len(df_c) >= 12 else "Grau I", "Grau III" if (max_v-min_v)/(2*vu) < 0.2 else "Grau I")
                
                # Exibição na Tela
                cols = st.columns(3)
                cols[0].metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                cols[1].metric("V.U. Médio", f"R$ {vu:,.2f}")
                cols[2].metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Res
