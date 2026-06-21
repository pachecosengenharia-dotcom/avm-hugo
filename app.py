import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
    # 1. Leitura forçada com tratamento de erro
    df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    df.columns = df.columns.str.strip()

    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    if features and target:
        # 2. BLINDAGEM DE DADOS: Remove tudo que não for número
        df_c = df[features + [target]].copy()
        for col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        df_c = df_c.dropna() # Remove linhas com dados faltantes
        
        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {}
            for f in features:
                inputs[f] = st.sidebar.number_input(f"{f}", value=float(df_c[f].median()))
            
            if st.sidebar.button("Calcular"):
                # Cálculo
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                
                # Resultados
                st.metric("V.U. Estimado", f"R$ {vu:,.2f}")
                
                # Gráfico
                fig, ax = plt.subplots()
                ax.scatter(df_c[target], modelo.predict(df_c[features]))
                st.pyplot(fig)
        else:
            st.error("Após a limpeza, não sobraram dados válidos. Verifique se o seu CSV possui números corretamente formatados.")
