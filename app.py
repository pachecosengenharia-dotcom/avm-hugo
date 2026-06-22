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

# Função para normalizar texto (limpeza de acentos para evitar problemas no celular)
def normalizar_texto(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def treinar_modelo_robusto(X, y):
    X = sm.add_constant(X)
    modelo = sm.OLS(y, X).fit()
    _, p_bp, _, _ = smd.het_breuschpagan(modelo.resid, modelo.model.exog)
    _, p_sw = shapiro(modelo.resid)
    
    # Autocorreção: Se falhar nos testes, aplica Log na variável alvo
    if p_bp < 0.05 or p_sw < 0.05:
        y_log = np.log(y)
        modelo = sm.OLS(y_log, X).fit()
        return modelo, True, p_bp, p_sw
    return modelo, False, p_bp, p_sw

def calcular_grau(modelo, n, p_bp, p_sw):
    ci = modelo.conf_int(alpha=0.20)
    amp = ((ci[1] - ci[0]) / (2 * modelo.predict(modelo.model.exog))).mean()
    sig = all(p < 0.05 for p in modelo.pvalues[1:])
    
    if amp <= 0.20 and sig and n >= 12 and p_bp > 0.05 and p_sw > 0.05:
        return "Grau III", amp
    elif amp <= 0.30 and n >= 6:
        return "Grau II", amp
    return "Grau I", amp

# Função para Gerar o PDF
def gerar_laudo_pdf(grau, amp, p_bp, p_sw):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação (NBR 14653-2)")
    c.setFont("Helvetica", 10)
    c.drawString(50, 750, f"Enquadramento: {grau} | Amplitude IC 80%: {amp:.2%}")
    c.drawString(50, 735, f"Teste Breusch-Pagan (p-valor): {p_bp:.4f}")
    c.drawString(50, 720, f"Teste Shapiro-Wilk (p-valor): {p_sw:.4f}")
    c.save()
    buffer.seek(0)
    return buffer

st.title("📊 AVM - Engenharia de Avaliações")
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type=["csv"])

if arquivo:
 df = pd.read_csv(arquivo)
    df.columns = [normalizar_texto(col).strip() for col in df.columns]
    target = st.sidebar.selectbox("Coluna Alvo:", options=df.columns)
    features = st.sidebar.multiselect("Variáveis Explicativas:", options=[c for c in df.columns if c != target])

    st.sidebar.header("📝 Dados do Imóvel")
    info = {
        "Endereço": st.sidebar.text_input("Endereço", key="info_end"),
        "Bairro": st.sidebar.text_input("Bairro", key="info_bairro"),
        "Informante": st.sidebar.text_input("Informante", key="info_inf")
    }

    if features and target:
        X, y = df[features], df[target]
        modelo, trans, p_bp, p_sw = treinar_modelo_robusto(X, y)
        grau, amp = calcular_grau(modelo, len(df), p_bp, p_sw)
        
        st.subheader("Resultados Técnicos")
        st.write(f"Classificação: **{grau}**")

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)

            inputs = {}
            st.sidebar.markdown("### Valores para Avaliação")
            for f in features:
                min_f, max_f = df_c[f].min(), df_c[f].max()
                val = st.sidebar.number_input(f"{f} (Min: {min_f:.2f} | Max: {max_f:.2f})", value=float(df_c[f].median()), key=f"inp_{f}")
                inputs[f] = val
                if val < min_f or val > max_f:
                    st.sidebar.warning(f"⚠️ Extrapolação em {f}!")

            if st.sidebar.button("Calcular Precificação", key="btn_calc"):
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
                
                # BOTÃO PARA BAIXAR O PDF
               if st.button("Gerar PDF"):
               pdf = gerar_laudo_pdf(grau, amp, p_bp, p_sw)
               st.download_button("📥 Baixar Laudo", pdf, "laudo_tecnico.pdf")
