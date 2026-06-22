import streamlit as st

import pandas as pd

import numpy as np

import io

import textwrap

import unicodedata

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4

from reportlab.pdfgen import canvas

from reportlab.lib.utils import ImageReader

def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# Configuração da Página

st.set_page_config(layout="wide", page_title="AVM - Engenharia de Avaliações")

st.title("📊 AVM - Engenharia de Avaliações")



# Função para Gerar o PDF

def gerar_laudo_pdf(d, fig, eq_str, info, graus, inputs, min_v, max_v):

    buffer = io.BytesIO()

    c = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4

    

    # Cabeçalho

    c.setFont("Helvetica-Bold", 16)

    c.drawString(50, height - 50, "Laudo Técnico de Avaliação (NBR 14653)")

    

    # Informações do Imóvel

    c.setFont("Helvetica-Bold", 12)

    c.drawString(50, height - 90, "Dados do Imóvel:")

    c.setFont("Helvetica", 10)

    y = height - 110

    for k, v in info.items():

        c.drawString(50, y, f"{k}: {v}")

        y -= 20

    

    # Equação do Modelo (com quebra de linha)

    y -= 10

    c.setFont("Helvetica-Bold", 12)

    c.drawString(50, y, "Equação do Modelo:")

    y -= 20

    c.setFont("Courier", 8)

    linhas_eq = textwrap.wrap(eq_str, width=100)

    for linha in linhas_eq:

        c.drawString(50, y, linha)

        y -= 12

    

    # Variáveis Utilizadas

    y -= 20

    c.setFont("Helvetica-Bold", 12)

    c.drawString(50, y, "Variáveis e Parâmetros Utilizados:")

    y -= 20

    c.setFont("Helvetica", 10)

    for k, v in inputs.items():

        c.drawString(50, y, f"- {k}: {v:.2f}")

        y -= 15

    

    # Resultados Finais

    y -= 20

    c.setFont("Helvetica-Bold", 12)

    c.drawString(50, y, "Resultados da Avaliação:")

    c.setFont("Helvetica", 10)

    y -= 20

    c.drawString(50, y, f"V.U. Mínimo: R$ {min_v:,.2f} | V.U. Médio: R$ {d['vu']:,.2f} | V.U. Máximo: R$ {max_v:,.2f}")

    y -= 20

    c.setFont("Helvetica-Bold", 12)

    c.drawString(50, y, f"VALOR TOTAL ESTIMADO: R$ {d['total']:,.2f}")

    y -= 20

    c.setFont("Helvetica", 10)

    c.drawString(50, y, f"Fundamentação: {graus[0]} | Precisão: {graus[1]}")

    

    # Inserção do Gráfico

    y -= 220

    img_buf = io.BytesIO()

    fig.savefig(img_buf, format='png', bbox_inches='tight')

    img_buf.seek(0)

    c.drawImage(ImageReader(img_buf), 50, y, width=400, height=200)

    

    c.showPage()

    c.save()

    buffer.seek(0)

    return buffer



# Interface Lateral e Lógica Principal

arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv", "txt"])



import unicodedata # Adicione este import no topo do seu arquivo

# Função para remover acentos dos nomes das colunas
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

if arquivo:
    try:
        df = pd.read_csv(arquivo, encoding='utf-8', sep=None, engine='python')
    except:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, encoding='latin-1', sep=None, engine='python')
    
    # Limpa nomes das colunas (isso remove os acentos e resolve o problema no celular)
    df.columns = [normalizar_texto(col).strip() for col in df.columns]

    
    # 3. Restante da lógica (target, features, etc)
    target = st.sidebar.selectbox(
        "Coluna Alvo:", 
        options=df.columns, 
        key="target_col"
    )
    
    features = st.sidebar.multiselect(
        "Variáveis Explicativas:", 
        options=[c for c in df.columns if c != target], 
        key="features_col"
    )
    

    st.sidebar.header("📝 Dados do Imóvel")

    info = {
        inputs = {}
    st.sidebar.markdown("### Valores para Avaliação")
    
    for f in features:
        min_f = df_c[f].min()
        max_f = df_c[f].max()
        
        # Cria o campo de entrada e mostra o intervalo permitido
        val = st.sidebar.number_input(
            f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", 
            value=float(df_c[f].median())
        )
        inputs[f] = val
        
        # Verificação de Extrapolação
        if val < min_f or val > max_f:
            st.sidebar.warning(f"⚠️ Extrapolação em {f}!")
    
    # Lógica do botão de cálculo
    if st.sidebar.button("Calcular Precificação", key="btn_calc"):
        # Converter inputs para array para o modelo
        vetor_input = np.array([list(inputs.values())])
        vu = modelo.predict(vetor_input)[0]

        "Endereço": st.sidebar.text_input("Endereço"),

        "Bairro": st.sidebar.text_input("Bairro"),

        "Informante": st.sidebar.text_input("Informante")

    }



    if features and target:

        df_c = df.copy()

        for col in features + [target]:

            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(',', '.'), errors='coerce')

        df_c = df_c.dropna()



        if not df_c.empty:

            modelo = LinearRegression().fit(df_c[features], df_c[target])

            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])

            st.latex(eq_str)

            # ... (linhas anteriores do seu código)
        st.latex(eq_str)
        
        # Bloco de inputs com verificação de extrapolação
        inputs = {}
        st.sidebar.markdown("### Valores para Avaliação")
        
        for f in features:
            min_f, max_f = df_c[f].min(), df_c[f].max()
            valor = st.sidebar.number_input(
                f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", 
                value=float(df_c[f].median()), 
                key=f"inp_{f}"
            )
            inputs[f] = valor
            if valor < min_f or valor > max_f:
                st.sidebar.warning(f"⚠️ Extrapolação em {f}!")
        
        # Botão de cálculo
        if st.sidebar.button("Calcular Precificação", key="btn_calc"):
            # ... (seu cálculo de modelo.predict continua aqui)

            inputs = {f: st.sidebar.number_input(f"{f}", value=float(df_c[f].median())) for f in features}
            # ... (dentro do seu bloco if not df_c.empty:)

        modelo = LinearRegression().fit(df_c[features], df_c[target])
        eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
        st.latex(eq_str)
        
        # O BLOCO DEVE COMEÇAR AQUI, SEM ESPAÇOS EXTRAS OU FALTANDO
        inputs = {}
        st.sidebar.markdown("### Valores para Avaliação")
        
        for f in features:
            min_f, max_f = df_c[f].min(), df_c[f].max()
            val = st.sidebar.number_input(
                f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", 
                value=float(df_c[f].median()), 
                key=f"inp_{f}"
            )
            inputs[f] = val
            if val < min_f or val > max_f:
                st.sidebar.warning(f"⚠️ Extrapolação em {f}!")

            

            if st.sidebar.button("Calcular Precificação"):

                vu = modelo.predict(np.array([list(inputs.values())]))[0]

                preds = modelo.predict(df_c[features])

                std = np.std(df_c[target] - preds)

                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)

                total = vu * inputs[features[0]] if features else vu

                graus = ("Grau III" if len(df_c) >= 12 else "Grau I", "Grau III" if (max_v-min_v)/(2*vu) < 0.2 else "Grau I")

                

                cols = st.columns(3)

                cols[0].metric("V.U. Mínimo", f"R$ {min_v:,.2f}")

                cols[1].metric("V.U. Médio", f"R$ {vu:,.2f}")

                cols[2].metric("V.U. Máximo", f"R$ {max_v:,.2f}")

                st.markdown(f"### Valor Total Estimado: R$ {total:,.2f}")

                

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

                ax1.scatter(df_c[target], preds); ax1.set_title("Aderência")

                ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")

                st.pyplot(fig)

                

                pdf = gerar_laudo_pdf({'vu': vu, 'total': total}, fig, eq_str, info, graus, inputs, min_v, max_v)

                st.download_button("📥 Baixar Laudo Completo", pdf, "laudo_tecnico.pdf")
