import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Função do Laudo PDF
def gerar_laudo_pdf(d, fig, eq_str, info):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 800, "Laudo Técnico de Avaliação")
    c.setFont("Helvetica", 10)
    y = 770
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15
    y -= 10
    c.drawString(50, y, f"V.U. Médio: R$ {d['vu']:,.2f} | Valor Total: R$ {d['total']:,.2f}")
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 400, width=400, height=200)
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
    df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis:", [c for c in df.columns if c != target])
    
    # Variáveis de texto
    st.sidebar.header("📝 Dados do Imóvel")
    info = {
        "Endereço": st.sidebar.text_input("Endereço"),
        "Complemento": st.sidebar.text_input("Complemento"),
        "Bairro": st.sidebar.text_input("Bairro"),
        "Informante": st.sidebar.text_input("Informante"),
        "Telefone": st.sidebar.text_input("Telefone")
    }

    if features and target:
        df_c = df.dropna(subset=features + [target])
        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq)
            
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                
                # Identifica a área para o valor total
                col_area = next((c for c in features if 'area' in c.lower()), features[0])
                total = vu * inputs[col_area]
                
                c1, c2 = st.columns(2)
                c1.metric("V.U. Estimado", f"R$ {vu:,.2f}")
                c2.metric("Valor Total Estimado", f"R$ {total:,.2f}")
                
                fig, ax = plt.subplots()
                ax.scatter(df_c[target], modelo.predict(df_c[features]))
                st.pyplot(fig)
                
                pdf = gerar_laudo_pdf({'vu': vu, 'total': total}, fig, eq, info)
                st.download_button("📥 Baixar Laudo", pdf, "laudo.pdf")
