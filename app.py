import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    # Correção do erro de leitura (uso de encoding='latin-1' resolve o UnicodeDecodeError)
    df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    df.columns = df.columns.str.strip()
    
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis:", [c for c in df.columns if c != target])
    
    if features and target:
        df_c = df.dropna(subset=features+[target])
        modelo = LinearRegression().fit(df_c[features], df_c[target])
        
        st.sidebar.header("⚙️ Parâmetros")
        inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
        
        if st.sidebar.button("Calcular"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            st.metric("V.U. Estimado", f"R$ {vu:,.2f}")
            
            # Gráficos simples
            fig, ax = plt.subplots()
            ax.scatter(df_c[target], modelo.predict(df_c[features]))
            st.pyplot(fig)
