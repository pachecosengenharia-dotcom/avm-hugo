import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (Completo)")

# --- 1. FUNÇÃO DE PDF ---
def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 9)
    y = 770
    c.drawString(50, y, "Equação:")
    y -= 15
    for i in range(0, len(eq_str), 100):
        c.drawString(50, y, eq_str[i:i+100]); y -= 12
    c.drawString(50, y-10, "Parâmetros Utilizados:")
    for k, v in inputs.items():
        c.drawString(60, y-25, f"- {k}: {v:.2f}"); y -= 12
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y-20, f"Resultados: V.U. Médio R$ {d['vu']:,.2f} | Total R$ {d['total']:,.2f}")
    c.drawString(50, y-35, f"Intervalo (95%): R$ {d['min']:,.2f} a R$ {d['max']:,.2f}")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=400, height=200)
    c.save(); buffer.seek(0)
    return buffer

# --- 2. CARREGAMENTO UNIVERSAL ---
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])
if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Coluna Valor Unitário:", cols)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in cols if c != target])
    
    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            # Gráficos
            preds = modelo.predict(df_c[features])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
            ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
            st.pyplot(fig)

            # Parâmetros com limites NBR
            st.sidebar.header("⚙️ Parâmetros (Limites)")
            inputs = {f: st.sidebar.number_input(f"{f} ({df_c[f].min():.1f} a {df_c[f].max():.1f})", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * (inputs.get('Área Privativa', 1))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
                
                pdf = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, eq_str, inputs)
                st.download_button("📥 Baixar Laudo PDF", pdf, "laudo_tecnico.pdf")
