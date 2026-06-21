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

def limpar_coluna(col):
    # Converte strings de moeda/formato brasileiro para float, mantendo o que já é número
    return pd.to_numeric(col.astype(str).str.replace(r'[^\d,.-]', '', regex=True).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar(c) for c in df.columns]
    
    # 1. Identificação de colunas com fallback
    target = next((c for c in df.columns if any(x in c for x in ['valor', 'preco', 'unitario'])), df.columns[-1])
    col_area = next((c for c in df.columns if 'area' in c), None)
    
    st.write(f"**Coluna Alvo detectada:** {target}")
    st.write(f"**Coluna Área detectada:** {col_area}")

    # 2. Limpeza rigorosa sem descartar linhas antes da hora
    df_c = df.copy()
    for col in df_c.columns:
        df_c[col] = limpar_coluna(df_c[col])
    
    # Agora removemos apenas se for estritamente necessário para o modelo
    features = st.sidebar.multiselect("Variaveis Explicativas:", [c for c in df_c.columns if c not in [target, col_area]])
    df_clean = df_c.dropna(subset=features + [target])

    if not df_clean.empty and features:
        modelo = LinearRegression().fit(df_clean[features], df_clean[target])
        
        # Exibição do Modelo
        st.latex(f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]))
        
        # Gráficos diagnósticos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        preds = modelo.predict(df_clean[features])
        ax1.scatter(df_clean[target], preds); ax1.set_title("Aderencia")
        ax2.scatter(preds, df_clean[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Residuos")
        st.pyplot(fig)

        # Inputs do Imóvel Avaliando
        st.sidebar.header("⚙️ Parametros")
        inputs = {f: st.sidebar.number_input(f"{f} (Mediana: {df_clean[f].median():.1f})", value=float(df_clean[f].median())) for f in features}
        area_aval = st.sidebar.number_input("Area do Imovel Avaliando:", value=float(df_clean[col_area].median()) if col_area else 1.0)
        
        if st.sidebar.button("Calcular"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            # Cálculo de Intervalo NBR 14653
            std = np.std(df_clean[target] - preds)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            
            st.metric("V.U. Estimado", f"R$ {vu:,.2f}")
            st.metric("Valor Total", f"R$ {vu * area_aval:,.2f}")
            
            # Mostra contagem de dados usados
            st.info(f"Modelo treinado com {len(df_clean)} amostras.")
