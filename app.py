import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Função para remover acentos e caracteres especiais
def limpar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# ... (funções calcular_graus e gerar_laudo_pdf permanecem iguais) ...

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])

if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    
    # LIMPEZA DE COLUNAS: Remove acentos dos nomes para evitar erro no celular
    df.columns = [limpar_texto(col.strip()) for col in df.columns]
    
    target = st.sidebar.selectbox("Selecionar Coluna Alvo:", df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in df.columns if c != target])
    
    # ... (info_extra e tratamento de dados iguais) ...

    if features and target:
        df_c = df.copy()
        # ... (limpeza de dados e modelo) ...
            
            st.sidebar.header("📊 Parâmetros de Entrada")
            # Nomes limpos aqui também
            inputs = {f: st.sidebar.number_input(f"{f} (Lim: {df_c[f].min():.1f} a {df_c[f].max():.1f})", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                # ... (cálculos de vu, min_v, max_v, graus) ...
                
                # Exibição otimizada para celular
                st.write("---")
                st.info(f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")
                
                # Colunas de métricas separadas
                col1, col2 = st.columns(2)
                col1.metric("V.U. Minimo", f"R$ {min_v:,.2f}")
                col2.metric("V.U. Maximo", f"R$ {max_v:,.2f}")
                st.metric("V.U. Medio", f"R$ {vu:,.2f}")
                
                # Destaque claro para o Valor Total
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                    <h3 style="margin:0;">Valor Total do Imóvel:</h3>
                    <h2 style="margin:0;">R$ {:,.2f}</h2>
                </div>
                """.format(total), unsafe_allow_html=True)
                
                # ... (botão de download) ...
