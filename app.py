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
        s = str(valor).replace('R$', '').replace('.', '').replace(',', '.')
        return float(s)
    except:
        return np.nan

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar(c) for c in df.columns]
    
    # Detecção automática ou Manual
    target = next((c for c in df.columns if any(x in c for x in ['valor', 'preco', 'unitario'])), df.columns[-1])
    # Se não achar a área, o usuário escolhe
    col_area = st.sidebar.selectbox("Coluna da Área Privativa:", df.columns, index=0)
    
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c not in [target, col_area]])

    st.sidebar.header("📝 Dados do Imóvel")
    info = {k: st.sidebar.text_input(k) for k in ["Endereço", "Bairro", "Informante"]}

    if features:
        df_clean = df.copy()
        df_clean[target] = df_clean[target].apply(limpar_numero)
        df_clean[col_area] = df_clean[col_area].apply(limpar_numero)
        for f in features: df_clean[f] = df_clean[f].apply(limpar_numero)
        df_clean = df_clean.dropna(subset=[target] + features)

        modelo = LinearRegression().fit(df_clean[features], df_clean[target])
        
        # Exibição do gráfico e modelo
        st.latex(f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]))
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(modelo.predict(df_clean[features]), df_clean[target])
        st.pyplot(fig)

        st.sidebar.header("⚙️ Parâmetros")
        inputs = {f: st.sidebar.number_input(f"{f} (Mediana:{df_clean[f].median():.1f})", value=float(df_clean[f].median())) for f in features}
        area_aval = st.sidebar.number_input("Área do Imóvel Avaliando (m2):", value=float(df_clean[col_area].median()))
        
        # O BOTÃO DE CALCULAR AGORA APARECE SEMPRE QUE HOUVER FEATURES
        if st.sidebar.button("Calcular Precificação"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            st.metric("V.U. Estimado", f"R$ {vu:,.2f}")
            st.metric("Valor Total Estimado", f"R$ {vu * area_aval:,.2f}")
            
            # (Adicione aqui a lógica de geração de PDF mantida anteriormente)
            st.success("Cálculo realizado com sucesso!")
    else:
        st.warning("Selecione as Variáveis Explicativas no menu lateral para habilitar o cálculo.")
