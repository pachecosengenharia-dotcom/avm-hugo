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
    c.drawString(50, y, "Equacao do Modelo:"); y -= 15
    for i in range(0, len(eq_str), 80):
        c.drawString(50, y, eq_str[i:i+80]); y -= 12
    y -= 10
    c.drawString(50, y, f"RESULTADOS: Min: R$ {d['min']:,.2f} | Medio: R$ {d['vu']:,.2f} | Max: R$ {d['max']:,.2f}")
    c.drawString(50, y-15, f"VALOR TOTAL ESTIMADO: R$ {d['total']:,.2f}")
    if fig is not None:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 50, y-220, width=400, height=180)
    c.save(); buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])
fig = None

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar(c) for c in df.columns]
    
    target = next((c for c in df.columns if any(x in c for x in ['valor', 'preco', 'unitario'])), df.columns[-1])
    col_area = next((c for c in df.columns if 'area' in c), None)
    
    features = st.sidebar.multiselect("Variaveis Explicativas:", [c for c in df.columns if c not in [target, col_area]])
    
    st.sidebar.header("📝 Dados do Imovel")
    info_imovel = {k: st.sidebar.text_input(k) for k in ["Endereco", "Bairro", "Informante"]}
    
    if features and target and col_area:
        df_c = df.copy()
        for col in features + [target, col_area]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.scatter(df_c[target], modelo.predict(df_c[features])); ax1.set_title("Aderencia")
            ax2.scatter(modelo.predict(df_c[features]), df_c[target] - modelo.predict(df_c[features])); ax2.axhline(0, color='red'); ax2.set_title("Residuos")
            st.pyplot(fig)

            st.sidebar.header("⚙️ Parametros")
            inputs = {f: st.sidebar.number_input(f"{f} (Min:{df_c[f].min():.1f} | Max:{df_c[f].max():.1f})", value=float(df_c[f].median())) for f in features}
            area_imovel = st.sidebar.number_input("Area do imovel avaliando:", value=float(df_c[col_area].median()))
            
            if st.sidebar.button("Calcular Precificacao"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - modelo.predict(df_c[features]))
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                total = vu * area_imovel
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Minimo", f"R$ {min_v:,.2f}")
                c2.metric("Medio", f"R$ {vu:,.2f}")
                c3.metric("Maximo", f"R$ {max_v:,.2f}")
                st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
                
                pdf = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, eq_str, inputs, info_imovel)
                st.download_button("📥 Baixar Laudo Profissional", pdf, "laudo_final.pdf")
