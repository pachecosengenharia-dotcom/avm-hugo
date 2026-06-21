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
st.title("📊 AVM - Engenharia de Avaliações")

def gerar_laudo_pdf(d, fig, eq_str, inputs, info_extra, variaveis_limites):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Cabeçalho
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 820, "Laudo Técnico de Avaliação (NBR 14653)")
    
    # Dados Informativos
    c.setFont("Helvetica", 10)
    y = 790
    for label, val in info_extra.items():
        c.drawString(50, y, f"{label}: {val}")
        y -= 15
        
    # Equação
    y -= 10
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Equação do Modelo:")
    c.setFont("Helvetica", 9)
    c.drawString(50, y-15, eq_str)
    
    # Limites e Variáveis
    y -= 40
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Variáveis e Limites Utilizados:")
    c.setFont("Helvetica", 9)
    y -= 15
    for var, val in inputs.items():
        lim = variaveis_limites[var]
        c.drawString(50, y, f"- {var}: {val:.2f} (Limites: {lim['min']:.2f} a {lim['max']:.2f})")
        y -= 12
        
    # Resultados
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, f"Resultado: V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    c.drawString(50, y-15, f"Intervalo de Confiança 95%: R$ {d['min']:,.2f} a R$ {d['max']:,.2f}")
    
    # Gráfico
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=450, height=220)
    
    c.save()
    buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = df.columns.str.strip()
    
    # Configuração do Modelo
    st.sidebar.header("⚙️ Configuração")
    target = st.sidebar.selectbox("Selecionar Coluna Alvo (Valor Unitário):", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    # Inputs de texto para o laudo
    st.sidebar.header("📝 Dados do Imóvel")
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
            
            # Gráficos
            preds = modelo.predict(df_c[features])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
            ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
            st.pyplot(fig)

            # Inputs do Usuário para cálculo
            st.sidebar.header("📊 Parâmetros de Entrada")
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            variaveis_limites = {f: {'min': df_c[f].min(), 'max': df_c[f].max()} for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                
                col_area = next((c for c in features if 'area' in c.lower() or 'área' in c.lower()), None)
                total = vu * inputs[col_area] if col_area else vu
                
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                pdf = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, eq_str, inputs, info_extra, variaveis_limites)
                st.download_button("📥
