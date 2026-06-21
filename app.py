import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import re

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Insira sua planilha de mercado e ajuste os parâmetros para calcular o valor de mercado.")

# Barra Lateral - Upload dos Dados
st.sidebar.header("📁 Base de Dados")
arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])

if arquivo_upload:
    try:
        df = pd.read_csv(arquivo_upload, delimiter=';', encoding='latin-1')
        if len(df.columns) <= 1:
            df = pd.read_csv(arquivo_upload, delimiter=',', encoding='latin-1')
    except:
        df = pd.read_csv(arquivo_upload, delimiter=',', encoding='latin-1')
        
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Mapeamento Inteligente de Colunas
    mapeamento_colunas = {}
    for col in df.columns:
        col_normalizada = col.lower().strip()
        if any(x in col_normalizada for x in ['prec', 'val', 'vlr', 'montante']): mapeamento_colunas[col] = 'Preco'
        elif any(x in col_normalizada for x in ['are', 'm2', 'm²', 'metrag', 'dimen']): mapeamento_colunas[col] = 'Area'
        elif any(x in col_normalizada for x in ['quart', 'dorm', 'quar']): mapeamento_colunas[col] = 'Quartos'
        elif any(x in col_normalizada for x in ['vag', 'garag']): mapeamento_colunas[col] = 'Vagas'
        elif any(x in col_normalizada for x in ['cons', 'estad']): mapeamento_colunas[col] = 'Conservacao'
        elif any(x in col_normalizada for x in ['setor', 'bairro', 'local', 'fator']): mapeamento_colunas[col] = 'Setor_Urbano'

    df = df.rename(columns=mapeamento_colunas)
    colunas_obrigatorias = ['Preco', 'Area', 'Setor_Urbano']
    
    if not all(c in df.columns for c in colunas_obrigatorias):
        st.error(f"Não mapeamos as colunas essenciais. Colunas na planilha: {list(df.columns)}")
    else:
        if 'Quartos' not in df.columns: df['Quartos'] = 2
        if 'Vagas' not in df.columns: df['Vagas'] = 1
        if 'Conservacao' not in df.columns: df['Conservacao'] = 2

        def limpar_numero(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt: txt = txt.replace('.', '')
            txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        for col in ['Preco', 'Area', 'Quartos', 'Vagas', 'Conservacao', 'Setor_Urbano']:
            df[col] = df[col].apply(limpar_numero)

        df['Quartos'] = df['Quartos'].fillna(2)
        df['Vagas'] = df['Vagas'].fillna(1)
        df['Conservacao'] = df['Conservacao'].fillna(2)
        df['Setor_Urbano'] = df['Setor_Urbano'].fillna(1.0)
        df = df.dropna(subset=['Preco', 'Area'])

        # Painel de Controle das Características do Imóvel avaliando
        st.sidebar.header("⚙️ Características do Imóvel")
        area_avaliando = st.sidebar.number_input("Área Útil (m²)", value=75.0, step=1.0)
        quartos_avaliando = st.sidebar.slider("Quantidade de Quartos", 1, 5, 2)
        vagas_avaliando = st.sidebar.slider("Vagas de Garagem", 0, 5, 1)
        conservacao_avaliando = st.sidebar.selectbox("Estado de Conservação", [1, 2, 3], format_func=lambda x: {1:"Regular", 2:"Bom", 3:"Excelente"}[x])
        setor_urbano_avaliando = st.sidebar.number_input("Fator de Bairro / Setor Urbano", value=1.0, step=0.1)

        if len(df) >= 3:
            # Modelo Regressivo Multifatorial
            X = df[['Area', 'Quartos', 'Vagas', 'Conservacao', 'Setor_Urbano']]
            y = df['Preco']
            modelo = LinearRegression().fit(X, y)
            
            dados_imovel = np.array([[area_avaliando, quartos_avaliando, vagas_avaliando, conservacao_avaliando, setor_urbano_avaliando]])
            preco_estimado = max(0, modelo.predict(dados_imovel)[0])
            r2_score = modelo.score(X, y)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Exibição dos Resultados (Métricas)
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de Mercado Estimado", f"R$ {preco_estimado:,.2f}")
            c2.metric("Intervalo Admissível (Mín/Máx)", f"R$ {limite_inferior:,.2f} a R$ {limite_superior:,.2f}")
            c3.metric("Precisão do Modelo (R²)", f"{f'{r2_score*100:.2f}%' if r2_score > 0 else 'N/A'}")

            # Gráfico Interativo
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.scatterplot(data=df, x='Area', y='Preco', color='#002d62', alpha=0.6, ax=ax, label="Amostras")
            ax.scatter([area_avaliando], [preco_estimado], color='#d9534f', s=150, marker='*', label="Avaliando")
            ax.set_title("Modelo de Regressão Linear")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Estruturação do PDF
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', dpi=200)
            img_buf.seek(0)
            
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("LAUDO DE AVALIAÇÃO TÉCNICA MERCADOLÓGICA", ParagraphStyle('T', fontSize=18, textColor=colors.HexColor('#002d62'), alignment=1)),
                Spacer(1, 15),
                Paragraph(f"<b>Área Proposta:</b> {area_avaliando} m² | <b>Quartos:</b> {quartos_avaliando} | <b>Vagas:</b> {vagas_avaliando}<br/><b>Valor de Mercado Inferido: R$ {preco_estimado:,.2f}</b>", styles['Normal']),
                Spacer(1, 15),
                Image(img_buf, width=400, height=180)
            ]
            doc.build(story)
            pdf_buf.seek(0)

            st.sidebar.markdown("---")
            st.sidebar.download_button(label="📥 Baixar Laudo Oficial (PDF)", data=pdf_buf, file_name="Laudo_Profissional.pdf", mime="application/pdf")
        else:
            st.warning("Dados insuficientes na planilha para gerar cálculos técnicos.")
else:
    st.info("💡 Por favor, faça o upload de uma planilha .csv na barra lateral para iniciar.")