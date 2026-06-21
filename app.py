import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
    # Leitura forçada em UTF-8 ou Latin-1 para evitar UnicodeDecodeError
    try:
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    # O modelo só deve rodar se houver variáveis e alvo selecionados
    if len(features) > 0 and target is not None:
        df_c = df.dropna(subset=features + [target])
        
        if not df_c.empty:
            modelo = LinearRegression()
            modelo.fit(df_c[features], df_c[target])
            
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {}
            for f in features:
                val = st.sidebar.number_input(f"{f}", value=float(df_c[f].median()))
                inputs[f] = val
            
            if st.sidebar.button("Calcular Precificação"):
                input_array = np.array([list(inputs.values())])
                vu = modelo.predict(input_array)[0]
                
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v = vu - (1.96 * std)
                max_v = vu + (1.96 * std)
                
                st.subheader("Resultados")
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].scatter(df_c[target], preds)
                ax[0].set_title("Aderencia")
                ax[1].scatter(preds, df_c[target] - preds)
                ax[1].axhline(0, color='red')
                ax[1].set_title("Residuos")
                st.pyplot(fig)
        else:
            st.warning("Dados insuficientes.")
    else:
        st.info("Selecione alvo e variáveis.")
else:
    st.info("Carregue o CSV.")
