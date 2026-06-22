import streamlit as st
import pandas as pd
import numpy as np
import io
import textwrap
import unicodedata
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- Funções Auxiliares ---
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def calcular_graus_nbr(n, k, var_relativa):
    # Fundamentação: NBR 14653 exige n >= 3k + 6 para Grau III
    if n >= (3 * k + 6):
        fundamentacao = "Grau III"
    elif n >= (2 * k + 4):
        fundamentacao = "Grau II"
    else:
        fundamentacao = "Grau I"
    
    # Precisão: Baseada na variação relativa (Campo de Arbítrio de 80%)
    if var_relativa <= 0.15:
        precisao = "Grau III"
    elif var_relativa <= 0.25:
        precisao = "Grau II"
    else:
        precisao = "Grau I"
        
    return fundamentacao, precisao

# --- Interface ---
st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações (Conforme NBR 14653)")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    df.columns = [normalizar_texto(col).strip() for col in df.columns]
    
    target = st.sidebar.selectbox("Coluna Alvo (Preço):", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    if features and target:
        df_c = df.dropna(subset=features + [target])
        
        # Treinamento do Modelo
        modelo = LinearRegression().fit(df_c[features], df_c[target])
        
        # Inputs e Bloqueio de Extrapolação
        st.sidebar.markdown("### Valores para Avaliação")
        inputs = {}
        extrapolou = False
        
        for f in features:
            min_f, max_f = df_c[f].min(), df_c[f].max()
            val = st.sidebar.number_input(f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", value=float(df_c[f].median()))
            inputs[f] = val
            if val < min_f or val > max_f:
                st.sidebar.error(f"❌ {f} fora da amostra!")
                extrapolou = True
        
        if st.sidebar.button("Calcular Precificação"):
            if extrapolou:
                st.error("Erro: A avaliação não pode ser concluída com variáveis fora do campo amostral (Proibido pela NBR 14653).")
            else:
                # Cálculos Estatísticos
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                residuos = df_c[target] - preds
                std_res = np.std(residuos)
                
                # Intervalo de 80% (t-student para n-k-1 graus de liberdade)
                t_score = stats.t.ppf(0.9, len(df_c) - len(features) - 1)
                erro_padrao = t_score * std_res
                min_v, max_v = vu - erro_padrao, vu + erro_padrao
                
                # Graus NBR
                var_relativa = (max_v - min_v) / (2 * vu)
                grau_f, grau_p = calcular_graus_nbr(len(df_c), len(features), var_relativa)
                
                # Exibição
                c1, c2, c3 = st.columns(3)
                c1.metric("V.U. Mínimo (80%)", f"R$ {min_v:,.2f}")
                c2.metric("V.U. Estimado", f"R$ {vu:,.2f}")
                c3.metric("V.U. Máximo (80%)", f"R$ {max_v:,.2f}")
                
                st.info(f"Fundamentação: **{grau_f}** | Precisão: **{precisao}**")

                # Gráficos
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.scatter(df_c[target], preds)
                ax1.set_title("Aderência (Observado x Estimado)")
                ax2.scatter(preds, residuos)
                ax2.axhline(0, color='red')
                ax2.set_title("Análise de Resíduos")
                st.pyplot(fig)
