import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# 1. Upload do Arquivo
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
    # Leitura segura do CSV
    try:
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # 2. Configuração do Modelo
    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    if features and target:
        # Preparação dos dados
        df_c = df.dropna(subset=features + [target])
        
        # Treinamento do Modelo (Ocorre apenas se houver dados)
        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            
            st.sidebar.header("⚙️ Parâmetros de Entrada")
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            
            # 3. Botão de Execução
            if st.sidebar.button("Calcular Precificação"):
                # Cálculo da Predição
                input_array = np.array([list(inputs.values())])
                vu = modelo.predict(input_array)[0]
                
                # Cálculo de erro e intervalo
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                
                # Exibição dos resultados na tela
                st.subheader("Resultados da Avaliação")
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                # Gráficos de diagnóstico
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds)
                ax1.set_title("Aderência")
                ax2.scatter(preds, df_c[target] - preds)
                ax2.axhline(0, color='red')
                ax2.set_title("Resíduos")
                st.pyplot(fig)
        else:
            st.warning("Dados insuficientes após limpeza.")
    else:
        st.info("Selecione a coluna alvo e as variáveis para começar.")
else:
    st.info("Por favor, carregue um arquivo CSV para iniciar.")
