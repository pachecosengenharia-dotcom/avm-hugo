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

# Função de PDF mantida, mas certifique-se de fechar o canvas corretamente
def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação")
    c.setFont("Helvetica", 9)
    y = 770
    c.drawString(50, y, "Equação:")
    y -= 15
    c.drawString(50, y, eq_str[:100]) # Simplificado para evitar erros
    c.drawString(50, y-20, "Parâmetros:")
    for k, v in inputs.items():
        y -= 12
        c.drawString(60, y, f"- {k}: {v:.2f}")
    
    # Adicionar Gráfico
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 100, width=400, height=200)
    c.save()
    buffer.seek(0)
    return buffer

# --- CARREGAMENTO E PROCESSAMENTO ---
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])
if arquivo:
    # Usar separador dinâmico
    df = pd.read_csv(arquivo, sep=None, engine='python')
    cols = df.columns.tolist()
    
    target = st.sidebar.selectbox("Coluna Valor Unitário:", cols)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in cols if c != target])
    
    if features and target:
        # Tratamento de dados mais robusto
        df_c = df.copy()
        for col in features + [target]:
            if df_c[col].dtype == 'object':
                df_c[col] = df_c[col].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
        df_c = df_c.dropna()

        # Modelo
        modelo = LinearRegression().fit(df_c[features], df_c[target])
        st.subheader("Equação do Modelo")
        st.latex(f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]))
        
        # Gráficos
        preds = modelo.predict(df_c[features])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")
        ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
        st.pyplot(fig)

        # Inputs
        st.sidebar.header("⚙️ Parâmetros")
        inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
        
        if st.sidebar.button("Calcular Precificação"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            std = np.std(df_c[target] - preds)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            
            # Cálculo do total (considera coluna 'Área' se existir)
            area_col = next((c for c in df_c.columns if 'área' in c.lower()), None)
            total = vu * (inputs.get(area_col, 1) if area_col else 1)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
            col2.metric("V.U. Médio", f"R$ {vu:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
            
            # Geração do PDF pós-cálculo
            pdf_data = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, "eq_str", inputs)
            st.download_button("📥 Baixar Laudo PDF", pdf_data, "laudo.pdf", "application/pdf")
