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

# Função de PDF Corrigida e Completa
def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, limites):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Técnico de Avaliação (NBR 14653)")
    c.setFont("Helvetica", 10)
    
    # Informações do Imóvel
    y = 790
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15
    
    # Equação
    y -= 10
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Equação do Modelo:")
    c.setFont("Helvetica", 9)
    c.drawString(50, y-15, eq_str[:90]) 
    
    # Variáveis e Limites
    y -= 40
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Variáveis e Limites Utilizados:")
    c.setFont("Helvetica", 9)
    y -= 15
    for f in inputs:
        lim = limites[f]
        c.drawString(50, y, f"- {f}: Valor {inputs[f]:.2f} (Limites: {lim['min']:.2f} a {lim['max']:.2f})")
        y -= 12
        
    # Resultados
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, f"V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    c.drawString(50, y-15, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
    
    # Gráfico
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=400, height=200)
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
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
        df_c = df_c.dropna(subset=features + [target])

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            st.sidebar.header("⚙️ Parâmetros")
            limites = {f: {'min': df_c[f].min(), 'max': df_c[f].max()} for f in features}
            inputs = {f: st.sidebar.number_input(f"{f} (Lim: {limites[f]['min']:.1f} a {limites[f]['max']:.1f})", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                graus = ("Grau III" if len(df_c) >= 12 else "Grau I", "Grau III" if (max_v-min_v)/(2*vu) < 0.2 else "Grau I")
                
                # Exibição
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
                st
