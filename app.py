import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

def calcular_enquadramento(modelo, n, k):
    """
    Lógica simplificada baseada no Anexo A da NBR 14653-2.
    """
    # 1. Significância (p-valor dos regressores < 0.05)
    p_values = modelo.pvalues[1:] 
    todos_significativos = all(p < 0.05 for p in p_values)
    
    # 2. Precisão (Amplitude do IC de 80%)
    ci = modelo.conf_int(alpha=0.20)
    amplitude = (ci[1] - ci[0]) / (2 * modelo.predict(modelo.model.exog))
    
    # Classificação (Exemplo de lógica)
    if n >= 12 and todos_significativos and amplitude.mean() < 0.20:
        return "Grau III"
    elif n >= 6 and amplitude.mean() < 0.30:
        return "Grau II"
    else:
        return "Grau I"

st.title("🛡️ AVM - Engenharia de Avaliações (NBR 14653-2)")

# Carregamento e Tratamento
arquivo = st.sidebar.file_uploader("Base de Dados (CSV)", type="csv")
if arquivo:
    df = pd.read_csv(arquivo)
    target = st.sidebar.selectbox("Variável Dependente (Preço):", df.columns)
    features = st.sidebar.multiselect("Variáveis Independentes:", [c for c in df.columns if c != target])

    if features and target:
        X = sm.add_constant(df[features]) # Necessário para o Intercepto
        y = df[target]
        
        # Ajuste do Modelo OLS (Ordinary Least Squares)
        modelo = sm.OLS(y, X).fit()
        
        # Exibição do Relatório Técnico
        st.subheader("Relatório de Significância Estatística")
        st.write(modelo.summary())
        
        # Cálculo de Graus
        grau = calcular_enquadramento(modelo, len(df), len(features))
        st.metric("Enquadramento Normativo", grau)
        
        # Testes de Normalidade e Homocedasticidade
        st.subheader("Validação de Pressupostos")
        residuos = modelo.resid
        col1, col2 = st.columns(2)
        col1.write("Teste de Normalidade (Shapiro-Wilk):")
        from scipy.stats import shapiro
        stat, p = shapiro(residuos)
        col1.write(f"p-valor: {p:.4f} ({'Normal' if p > 0.05 else 'Não Normal'})")
