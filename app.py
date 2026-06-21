import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# 1. UPLOAD DO ARQUIVO
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo is not None:
    # Leitura robusta
    try:
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    target = st.sidebar.selectbox("Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    # Inputs de texto (apenas exibição)
    st.sidebar.header("📝 Dados do Imóvel")
    st.sidebar.text_input("Endereço")
    st.sidebar.text_input("Complemento")
    st.sidebar.text_input("Bairro")
    st.sidebar.text_input("Informante")

    # 2. SE O USUÁRIO SELECIONOU AS VARIÁVEIS
    if features and target:
        # Limpeza total: força numérico e remove NaNs
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna(subset=features + [target])

        if not df_c.empty:
            # TREINAMENTO SÓ ACONTECE AQUI
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            
            # 3. CÁLCULO SÓ ACONTECE NO CLIQUE DO BOTÃO
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                
                # Estatísticas
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                min_v = vu - (1.96 * std)
                max_v = vu + (1.96 * std)
                
                # Exibição
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
                
                fig, ax = plt.subplots()
                ax.scatter(df_c[target], preds)
                st.pyplot(fig)
        else:
            st.warning("Dados insuficientes após limpeza.")
    else:
        st.info("Selecione a coluna alvo e as variáveis.")
