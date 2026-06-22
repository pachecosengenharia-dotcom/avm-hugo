import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Função PDF ---
def gerar_laudo_pdf(vu, total, grau_f, grau_p, inputs, min_v, max_v):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO - NBR 14653")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Valor Unitário Estimado: R$ {vu:,.2f}")
    c.drawString(50, 750, f"VALOR TOTAL ESTIMADO: R$ {total:,.2f}")
    c.drawString(50, 730, f"Fundamentação: {grau_f} | Precisão: {grau_p}")
    c.save()
    buffer.seek(0)
    return buffer

# --- Configuração ---
st.set_page_config(layout="wide", page_title="AVM - Engenharia")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin-1')
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox("Coluna Alvo (Preço):", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    # ... após o carregamento do arquivo ...
    if features and target:
        # Cria uma cópia limpa do DataFrame
        df_c = df.copy()
        
        # 1. FORÇA CONVERSÃO NUMÉRICA E REMOVE O QUE NÃO FOR NÚMERO
        cols_usadas = features + [target]
        for col in cols_usadas:
            # Substitui vírgula por ponto, converte pra número e força NaN onde der erro
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # 2. REMOVE LINHAS VAZIAS (ESSENCIAL PARA O SKLEARN)
        df_c = df_c.dropna(subset=cols_usadas)
        
        # Verifica se sobrou amostra válida
        if len(df_c) < 3:
            st.error(f"Erro: Sobraram apenas {len(df_c)} dados válidos após a limpeza. Verifique se o CSV tem valores numéricos nas colunas selecionadas.")
        else:
            # 3. AGORA O FIT FUNCIONA SEM ERRO
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            
            # ... (seu código de input e botão continua aqui) ...
        if st.sidebar.button("Calcular Precificação", key="btn_calc"):
            if extrapolou:
                st.error("Erro: Variáveis fora do limite amostral!")
            else:
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                area = list(inputs.values())[0]
                total = vu * area
                
                # Cálculos NBR
                preds = modelo.predict(df_c[features])
                residuos = df_c[target] - preds
                t_score = stats.t.ppf(0.9, len(df_c) - len(features) - 1)
                margem = t_score * np.std(residuos)
                
                # Exibição
                st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
                
                # PDF
                pdf_buffer = gerar_laudo_pdf(vu, total, "Grau III", "Grau III", inputs, vu-margem, vu+margem)
                st.download_button("📥 Baixar Laudo Completo (PDF)", pdf_buffer, "laudo.pdf")
