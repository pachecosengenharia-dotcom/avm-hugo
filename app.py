import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Sidebar for file upload
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
    # 1. READ AND CLEAN DATA
    try:
        # Use latin-1 to handle typical Brazilian CSV formatting (commas/accents)
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()

    # 2. SELECTION
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    # 3. ONLY PROCEED IF FEATURES AND TARGET ARE SELECTED
    if features and target:
        # Data Cleaning: Coerce to numeric, drop invalid rows
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna(subset=features + [target])

        if not df_c.empty:
            # Model Training
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            
            # Sidebar Inputs
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {}
            for f in features:
                min_val, max_val = float(df_c[f].min()), float(df_c[f].max())
                inputs[f] = st.sidebar.number_input(f"{f} (Lim: {min_val:.1f} a {max_val:.1f})", 
                                                    value=float(df_c[f].median()))
            
            # 4. EXECUTION BUTTON
            if st.sidebar.button("Calcular Precificação"):
                input_array = np.array([list(inputs.values())])
                vu = modelo.predict(input_array)[0]
                
                # Stats
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                
                # Display Results
                st.subheader("Resultados")
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                # Visuals
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds)
                ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds)
                ax2.axhline(0, color='red')
                ax2.set_title("Resíduos")
                st.pyplot(fig)
        else:
            st.warning("Dados insuficientes ou inválidos após limpeza.")
    else:
        st.info("Selecione a coluna alvo e as variáveis para treinar o modelo.")
else:
    st.info("Por favor, carregue um arquivo CSV para começar.")
