import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

# --- Configuração ---
st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

# --- Funções de Cálculo NBR ---
def calcular_graus_nbr(n, k, var_relativa):
    fund = "Grau III" if n >= (3 * k + 6) else ("Grau II" if n >= (2 * k + 4) else "Grau I")
    prec = "Grau III" if var_relativa <= 0.15 else ("Grau II" if var_relativa <= 0.25 else "Grau I")
    return fund, prec

# --- Interface ---
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    try:
        df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin-1')
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()

    target = st.sidebar.selectbox("Coluna Alvo (Preço):", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    if features and target:
        # Limpeza e conversão forçada para numérico
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        df_c = df_c.dropna(subset=features + [target])

        if len(df_c) < (len(features) + 2):
            st.warning("Dados insuficientes para regressão linear.")
        else:
            # Treinamento
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            
            # Entradas do usuário
            st.sidebar.markdown("### Valores para Avaliação")
            inputs = {}
            extrapolou = False
            for f in features:
                min_f, max_f = df_c[f].min(), df_c[f].max()
                val = st.sidebar.number_input(f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", value=float(df_c[f].median()))
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.error(f"⚠️ {f} fora do campo amostral!")
                    extrapolou = True
            
           # ... (código anterior de treino do modelo) ...

            # Botão de Cálculo
          # ... (código anterior de treino do modelo) ...

            # Botão de Cálculo
            # --- Inputs com KEYS únicas ---
            st.sidebar.markdown("### Valores para Avaliação")
            inputs = {}
            for f in features:
                min_f, max_f = df_c[f].min(), df_c[f].max()
                # Adicionamos key={f} para garantir unicidade
                val = st.sidebar.number_input(
                    f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", 
                    value=float(df_c[f].median()), 
                    key=f"input_{f}" 
                )
                inputs[f] = val
            
            # --- Botão com KEY única ---
            if st.sidebar.button("Calcular Precificação", key="btn_calcular_precificacao"):
                # ... (resto da sua lógica de cálculo) ...
def gerar_laudo_pdf(vu, total, info, grau_f, grau_p, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO - NBR 14653")
    c.drawString(50, 780, f"Valor Unitário Estimado: R$ {vu:,.2f}")
    c.drawString(50, 760, f"VALOR TOTAL ESTIMADO: R$ {total:,.2f}")
    c.drawString(50, 740, f"Fundamentação: {grau_f} | Precisão: {grau_p}")
    c.save()
    buffer.seek(0)
    return buffer

# --- Dentro do Botão "Calcular Precificação" ---
if st.sidebar.button("Calcular Precificação"):
    # ... (cálculos anteriores) ...
    
    # 1. Cálculo do Valor Total (Assumindo que a primeira feature é a Área Principal)
    area_principal = list(inputs.values())[0] 
    total = vu * area_principal
    
    # 2. Exibição
    st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
    
    # 3. Botão de Download do PDF
    info_dict = {"Endereço": "Logradouro A", "Cidade": "Goiânia"} # Ajuste conforme necessário
    pdf_buffer = gerar_laudo_pdf(vu, total, info_dict, grau_f, grau_p, inputs, min_v, max_v)
    
    st.download_button(
        label="📥 Baixar Laudo Completo (PDF)",
        data=pdf_buffer,
        file_name="laudo_tecnico.pdf",
        mime="application/pdf"
    )
