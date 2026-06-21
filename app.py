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
st.markdown("""<meta charset="UTF-8">""", unsafe_allow_html=True)
st.title("📊 AVM - Engenharia de Avaliações")

def calcular_graus(n, n_vars, intervalo_relativo):
    fund = "Grau I" if n < 6*n_vars else ("Grau II" if n < 12*n_vars else "Grau III")
    prec = "Grau I" if intervalo_relativo > 0.30 else ("Grau II" if intervalo_relativo > 0.20 else "Grau III")
    return fund, prec

def gerar_laudo_pdf(d, fig, eq_str, inputs, info_extra, variaveis_limites, graus):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    y = 790
    for label, val in info_extra.items():
        c.drawString(50, y, f"{label}: {val}")
        y -= 15
    y -= 10
    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    y -= 25
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Equação e Limites:")
    c.setFont("Helvetica", 9)
    y -= 15
    c.drawString(50, y, eq_str[:100])
    y -= 20
    for var, val in inputs.items():
        lim = variaveis_limites.get(var, {'min': 0, 'max': 0})
        c.drawString(50, y, f"- {var}: {val:.2f} (Limites: {lim['min']:.2f} a {lim['max']:.2f})")
        y -= 12
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, f"V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    c.drawString(50, y-15, f"Intervalo 95%: R$ {d['min']:,.2f} a R$ {d['max']:,.2f}")
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=400, height=200)
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Selecionar Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    info_extra = {
        "Endereço": st.sidebar.text_input("Endereço"),
        "Bairro": st.sidebar.text_input("Bairro"),
        "Informante": st.sidebar.text_input("Informante/Tel")
    }

    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            st.sidebar.header("📊 Parâmetros de Entrada")
            inputs = {f: st.sidebar.number_input(f"{f} (Limites: {df_c[f].min():.1f} a {df_c[f].max():.1f})", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                preds = modelo.predict(df_c[features])
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
                st.pyplot(fig)
                
                variaveis_limites = {f: {'min': df_c[f].min(), 'max': df_c[f].max()} for f in features}
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                
                intervalo_relativo = (max_v - min_v) / (2 * vu)
                graus = calcular_graus(len(df_c), len(features), intervalo_relativo)
                
                col_area = next((c for c in features if 'area' in c.lower() or 'área' in c.lower()), None)
                total = vu * inputs[col_area] if col_area else vu
                
                st.info(f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")
                
                pdf = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, eq_str, inputs, info_extra, variaveis_limites, graus)
                st.download_button("Baixar Laudo Técnico", pdf, "laudo_tecnico.pdf")
