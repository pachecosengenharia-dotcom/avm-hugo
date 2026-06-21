import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import unicodedata
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def normalizar(texto):
    return unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def limpar_numero(valor):
    try:
        # Remove R$, espaços e pontos de milhar, troca vírgula por ponto
        s = str(valor).replace('R$', '').replace('.', '').replace(',', '.')
        return float(s)
    except:
        return np.nan

def gerar_laudo_pdf(d, fig, eq_str, inputs, info_imovel):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Tecnico (NBR 14653)")
    c.setFont("Helvetica", 10)
    
    y = 790
    for k, v in info_imovel.items():
        c.drawString(50, y, f"{k}: {v}"); y -= 15
        
    y -= 10
    c.drawString(50, y, "Equacao do Modelo:"); y -= 12
    c.drawString(50, y-15, eq_str[:90])
    
    y -= 40
    c.drawString(50, y, f"V.U. Estimado: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y-200, width=400, height=180)
    c.save(); buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])
if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar(c) for c in df.columns]

    # Detecta automaticamente colunas de Valor e Área
    col_valor = next((c for c in df.columns if any(x in c for x in ['valor', 'preco', 'unitario'])), df.columns[-1])
    col_area = next((c for c in df.columns if 'area' in c), df.columns[0])
    
    st.sidebar.info(f"Base: {len(df)} linhas | Alvo: {col_valor}")
    features = st.sidebar.multiselect("Variaveis Explicativas:", [c for c in df.columns if c not in [col_valor, col_area]])

    st.sidebar.header("📝 Dados do Imovel")
    info = {k: st.sidebar.text_input(k) for k in ["Endereco", "Bairro", "Informante"]}

    if features and col_area:
        df_clean = df.copy()
        df_clean[col_valor] = df_clean[col_valor].apply(limpar_numero)
        df_clean[col_area] = df_clean[col_area].apply(limpar_numero)
        for f in features: df_clean[f] = df_clean[f].apply(limpar_numero)
        df_clean = df_clean.dropna()

        modelo = LinearRegression().fit(df_clean[features], df_clean[col_valor])
        eq_str = f"{col_valor} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
        st.latex(eq_str)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        preds = modelo.predict(df_clean[features])
        ax1.scatter(df_clean[col_valor], preds); ax1.set_title("Aderencia")
        ax2.scatter(preds, df_clean[col_valor] - preds); ax2.axhline(0, color='red'); ax2.set_title("Residuos")
        st.pyplot(fig)

        inputs = {f: st.sidebar.number_input(f"{f} (Mediana:{df_clean[f].median():.1f})", value=float(df_clean[f].median())) for f in features}
        area_aval = st.sidebar.number_input("Area do imovel avaliando:", value=float(df_clean[col_area].median()))

        if st.sidebar.button("Calcular Precificacao"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            st.metric("V.U. Estimado", f"R$ {vu:,.2f}")
            st.metric("Valor Total Estimado", f"R$ {vu * area_aval:,.2f}")
            
            pdf = gerar_laudo_pdf({'vu': vu, 'total': vu * area_aval}, fig, eq_str, inputs, info)
            st.download_button("📥 Baixar Laudo Profissional", pdf, "laudo.pdf")
