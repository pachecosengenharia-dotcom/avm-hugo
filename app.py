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

# Função para normalizar texto (limpeza de acentos para o PDF)
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# Função para Gerar o PDF
def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Laudo Tecnico de Avaliacao (NBR 14653)")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 90, "Dados do Imovel:")
    c.setFont("Helvetica", 10)
    y = height - 110
    for k, v in info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20
    
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Equacao do Modelo:")
    y -= 20
    c.setFont("Courier", 8)
    linhas_eq = textwrap.wrap(eq_str, width=100)
    for linha in linhas_eq:
        c.drawString(50, y, linha)
        y -= 12
    
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Variaveis e Parametros Utilizados:")
    y -= 20
    c.setFont("Helvetica", 10)
    for k, v in inputs.items():
        c.drawString(50, y, f"- {k}: {v:.2f}")
        y -= 15
        
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Resultados da Avaliacao:")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(50, y, f"V.U. Minimo: R$ {min_v:,.2f} | V.U. Medio: R$ {d['vu']:,.2f} | V.U. Maximo: R$ {max_v:,.2f}")
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"VALOR TOTAL ESTIMADO: R$ {d['total']:,.2f}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Fundamentacao: {graus[0]} | Precisao: {graus[1]}")
    
    # Inserção do gráfico
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, y-220, width=400, height=200)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding='utf-8-sig')
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

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

            inputs = {}
            for f in features:
                # Lógica de Limites: Trava de 50-1500 para Setor
                if "setor" in f.lower():
                    min_f, max_f = 50.0, 1500.0
                else:
                    min_f, max_f = df_c[f].min(), df_c[f].max()
                
                val = st.sidebar.number_input(f"{f} (Lim: {min_f:.2f} | {max_f:.2f})", value=float(df_c[f].median()))
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.warning(f"⚠️ Extrapolação em {f}!")

            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * inputs[features[0]] if features else vu
                
                # NBR 14653: Critérios Básicos para Fundamentação e Precisão
                n = len(df_c)
                k = len(features)
                fund = "Grau III" if n >= 3*k else "Grau I"
                amp = (max_v - min_v) / (2 * vu)
                prec = "Grau III" if amp <= 0.2 else "Grau I"
                
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")
                st.write(f"**Fundamentação:** {fund} | **Precisão:** {prec}")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
                st.pyplot(fig)
                
                pdf = gerar_laudo_pdf({'vu': vu, 'total': total}, fig, eq_str, info, (fund, prec), inputs, min_v, max_v)
                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
