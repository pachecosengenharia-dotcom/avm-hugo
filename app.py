import io
import re
import numpy as np
import pandas as pd
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

st.set_page_config(page_title="Plataforma AVM SaaS - Híbrido Definitivo", page_icon="🏢", layout="wide")

# =====================================================================
# GERADOR DE PDF CUSTOMIZADO
# =====================================================================
def gerar_laudo_pdf_ia(tenant, variavel_alvo, valores, r2, n_amostras, status_juridico, score_juridico):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#1A365D"), spaceAfter=15)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#2B6CB0"), spaceAfter=8)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=6)

    story.append(Paragraph(f"LAUDO CORE AVM - INTELIGÊNCIA ARTIFICIAL", title_style))
    story.append(Paragraph(f"<b>Instituição Solicitante:</b> {tenant}", text_style))
    story.append(Paragraph(f"<b>Variável Alvo Precificada:</b> {variavel_alvo.upper()}", text_style))
    story.append(Paragraph("<b>Metodologia Core:</b> Random Forest Regressor | NBR 14653", text_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Resultados do Motor de Machine Learning", subtitle_style))
    t2 = Table([
        ["Métrica de Cobertura do Risco", "Valor Comercial Admissível"],
        ["Margem Mínima de Segurança", f"R$ {valores['v_min']:,.2f}"],
        ["Valor de Face Estimado (Média)", f"R$ {valores['v_medio']:,.2f}"],
        ["Limite de Mercado Máximo", f"R$ {valores['v_max']:,.2f}"],
    ], colWidths=[260, 260])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"<b>Métricas do Modelo:</b> Precisão R² = {r2} | Amostras Saneadas = {n_amostras}.", text_style))

    story.append(Paragraph("2. Status da Esteira de Risco Jurídico", subtitle_style))
    t3 = Table([
        ["Status Documental", "APROVADO" if status_juridico else "REPROVADO"],
        ["Grau de Risco Legal", score_juridico],
    ], colWidths=[260, 260])
    t3.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor("#38A169") if status_juridico else colors.HexColor("#E53E3E")),
    ]))
    story.append(t3)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================================
# FUNÇÃO DE EXTRAÇÃO DE DADOS DE DOCUMENTOS (PDF + OCR)
# =====================================================================
def extrair_variaveis_de_documento(arquivo_pdf):
    texto_extraido = ""
    try:
        bytes_arquivo = arquivo_pdf.read()
    except Exception:
        return {}, ""
    
    try:
        with pdfplumber.open(io.BytesIO(bytes_arquivo)) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_extraido += texto + "\n"
    except Exception:
        pass

    if not texto_extraido.strip():
        try:
            imagens = convert_from_bytes(bytes_arquivo)
            for img in imagens:
                texto_ocr = pytesseract.image_to_string(img, lang='por')
                texto_extraido += texto_ocr + "\n"
        except Exception:
            pass

    if not texto_extraido.strip():
        return {}, ""

    variaveis_encontradas = {}
    trecho_limpo = texto_extraido.replace('\n', ' ')

    # 1. Área Privativa Coberta
    match_privativa = re.search(r'([\d.,]+)\s*metros\s*quadrados\s*de\s*área\s*privativa', trecho_limpo, re.IGNORECASE)
    if match_privativa:
        val = match_privativa.group(1).replace('.', '').replace(',', '.').replace('!', '1')
        try:
            variaveis_encontradas['area_privativa'] = float(val)
        except ValueError:
            pass

    # 2. Área do Terreno Relativa à Fração
    match_terreno_fracao = re.search(r'com\s*área\s*total\s*de\s*([\d.,]+)\s*metros\s*quadrados.*?fração', trecho_limpo, re.IGNORECASE)
    if not match_terreno_fracao:
        match_terreno_fracao = re.search(r'área\s*total\s*de\s*([\d.,]+)\s*metros\s*quadrados', trecho_limpo, re.IGNORECASE)
    
    if match_terreno_fracao:
        val = match_terreno_fracao.group(1).replace('.', '').replace(',', '.').replace('!', '1')
        try:
            variaveis_encontradas['area_terreno'] = float(val)
        except ValueError:
            pass

    # Isola o trecho específico da divisão interna para evitar conflitos com números de área
    match_divisao = re.search(r'divisão\s*interna[:\s]*(.*?)(?:edificada|lote|$)', trecho_limpo, re.IGNORECASE)
    trecho_divisao = match_divisao.group(1) if match_divisao else trecho_limpo

    # 3. Quartos
    match_quartos = re.search(r'(\d+)\s*\([^)]+\)\s*quartos', trecho_divisao, re.IGNORECASE)
    if not match_quartos:
        match_quartos = re.search(r'(\d+)\s*quarto[s]?', trecho_divisao, re.IGNORECASE)
    if match_quartos:
        try:
            variaveis_encontradas['quartos'] = int(match_quartos.group(1))
        except ValueError:
            pass

    # 4. Suítes
    match_suites = re.search(r'sendo\s*(\d+)', trecho_divisao, re.IGNORECASE)
    if not match_suites:
        match_suites = re.search(r'(\d+)\s*sui?te', trecho_divisao, re.IGNORECASE)
        
    val_suites = 0
    if match_suites:
        try:
            val_suites = int(match_suites.group(1))
        except ValueError:
            pass
    variaveis_encontradas['suites'] = val_suites
    variaveis_encontradas['suite'] = val_suites

    # 5. Banheiros
    match_banheiros = re.search(r'(\d+)\s*\([^)]+\)\s*banho', trecho_divisao, re.IGNORECASE)
    if not match_banheiros:
        match_banheiros = re.search(r'(\d+)\s*banho', trecho_divisao, re.IGNORECASE)
        
    val_banheiros = 1
    if match_banheiros:
        try:
            val_banheiros = int(match_banheiros.group(1))
        except ValueError:
            pass
    variaveis_encontradas['banheiros'] = val_banheiros

    # 6. Vagas
    match_vagas = re.search(r'(\d+)\s*\([^)]+\)\s*garagem', trecho_limpo, re.IGNORECASE)
    if not match_vagas:
        match_vagas = re.search(r'(\d+)\s*vaga[s]?', trecho_limpo, re.IGNORECASE)
    if match_vagas:
        try:
            variaveis_encontradas['vagas_garagem'] = int(match_vagas.group(1))
        except ValueError:
            pass

    return variaveis_encontradas, trecho_limpo[:600]

# =====================================================================
# INTERFACE PRINCIPAL DO PAINEL SAAS
# =====================================================================
st.title("🏢 Painel de Crédito e Controle AVM - Híbrido Definitivo")
st.markdown("Plataforma integrada para carga de mercado, leitura documental por IA e preenchimento paramétrico.")
st.divider()

st.sidebar.header("🔑 Assinatura e Faturamento")
tenant_selecionado = st.sidebar.selectbox("Cliente Institucional", ["001 - Banco Alfa S.A.", "002 - Imobiliária Local Ltda"])
plano_assinatura = "ENTERPRISE" if "Alfa" in tenant_selecionado else "STANDARD"
st.sidebar.markdown(f"**Plano Contratado:** {'🟢 ENTERPRISE' if plano_assinatura == 'ENTERPRISE' else '🟡 STANDARD'}")

aba_avm, aba_juridico = st.tabs([
    "📊 1. Carga, Leitura de Certidão & AVM Híbrido", 
    "📜 2. Análise Jurídica"
])

if 'status_juridico_global' not in st.session_state:
    st.session_state.status_juridico_global = True
if 'score_juridico_global' not in st.session_state:
    st.session_state.score_juridico_global = "PENDENTE"
if 'dados_extraidos_ia' not in st.session_state:
    st.session_state.dados_extraidos_ia = {}
if 'valores_manuais' not in st.session_state:
    st.session_state.valores_manuais = {}

with aba_avm:
    st.subheader("📁 1. Entradas de Dados: Planilha de Mercado & Certidão do Imóvel")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        arquivo_planilha = st.file_uploader("Envie a Base Comparativa (.xlsx ou .csv)", type=["xlsx", "csv"])
    with col_up2:
        documento_enviado = st.file_uploader("Envie a Certidão / Matrícula em PDF", type=["pdf"])

    if documento_enviado is not None:
        dados_extraidos, _ = extrair_variaveis_de_documento(documento_enviado)
        if dados_extraidos:
            st.session_state.dados_extraidos_ia = dados_extraidos
            # Limpa o cache manual anterior para forçar a nova leitura da certidão
            st.session_state.valores_manuais = {}
            st.success(f"✨ Certidão lida com sucesso! Variáveis extraídas automaticamente: {list(dados_extraidos.keys())}")

    df_global = None
    if arquivo_planilha is not None:
        try:
            if arquivo_planilha.name.endswith('.csv'):
                df_global = pd.read_csv(arquivo_planilha, encoding='latin1', sep=None, engine='python', on_bad_lines='skip')
            else:
                df_global = pd.read_excel(arquivo_planilha)
            
            df_global.columns = [
                c.lower().strip().replace(" ", "_")
                .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
                .replace("ã", "a").replace("õ", "o").replace("ç", "c").replace("â", "a").replace("ê", "e")
                for c in df_global.columns
            ]
            st.success(f"✅ Base de mercado processada com sucesso! {len(df_global)} linhas carregadas.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    else:
        data_padrao = {
            'valor_total_declarado': [450000, 480000, 510000, 750000, 820000, 350000],
            'area_privativa': [75.0, 78.0, 80.0, 85.0, 92.0, 60.0],
            'area_terreno': [200.0, 220.0, 250.0, 360.0, 400.0, 0.0],
            'quartos': [2, 2, 3, 3, 3, 1],
            'suites': [1, 1, 1, 2, 2, 0],
            'banheiros': [1, 1, 2, 2, 2, 1],
            'vagas_garagem': [1, 2, 2, 2, 3, 1],
            'indice_fiscal': [1200.0, 1250.0, 1300.0, 3200.0, 3300.0, 1500.0],
            'estado_conservacao': [3.0, 4.0, 3.0, 5.0, 4.0, 3.0]
        }
        df_global = pd.DataFrame(data_padrao)
        st.info("ℹ️ Utilizando base de dados padrão demonstrativa.")

    st.markdown("---")
    st.subheader("🤖 2. Configuração e Seleção de Variáveis Independentes")
    
    colunas_numericas = df_global.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(colunas_numericas) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            variavel_alvo = st.selectbox("Selecione a Variável Alvo (Preço/Valor):", colunas_numericas)
        with c2:
            features_disponiveis = [c for c in colunas_numericas if c != variavel_alvo]
            features_selecionadas = st.multiselect(
                "Escolha as Variáveis Independentes do Modelo (Auto + Manuais):",
                options=features_disponiveis,
                default=features_disponiveis[:min(5, len(features_disponiveis))]
            )

        if features_selecionadas:
            st.markdown("##### 📝 3. Atributos do Imóvel Avaliendo (Auto-preenchidos pela Certidão ou Ajustados Manualmente)")
            
            dados_ia = st.session_state.get('dados_extraidos_ia', {})
            campos_inteiros = ['quartos', 'suites', 'suite', 'banheiros', 'vagas', 'vagas_garagem', 'garagem']
            
            valores_usuario = {}
            cols_inputs = st.columns(len(features_selecionadas))
            
            for i, feat in enumerate(features_selecionadas):
                with cols_inputs[i % len(cols_inputs)]:
                    eh_inteiro = any(ci in feat.lower() for ci in campos_inteiros)
                    
                    # Define o valor inicial de forma segura usando o session_state de controle
                    if feat in st.session_state.valores_manuais:
                        val_inicial = st.session_state.valores_manuais[feat]
                    else:
                        val_inicial = float(df_global[feat].mean()) if not df_global[feat].empty else 0.0
                        for chave_ia, valor_ia in dados_ia.items():
                            if chave_ia == feat or chave_ia in feat or feat in chave_ia:
                                val_inicial = valor_ia
                                break
                    
                    if eh_inteiro:
                        val_inicial = int(round(float(val_inicial)))
                        valores_usuario[feat] = st.number_input(
                            f"{feat.replace('_', ' ').title()}", 
                            value=val_inicial,
                            step=1, 
                            format="%d",
                            key=f"input_safe_{feat}"
                        )
                    else:
                        val_inicial = float(val_inicial)
                        valores_usuario[feat] = st.number_input(
                            f"{feat.replace('_', ' ').title()}", 
                            value=val_inicial,
                            format="%.2f",
                            key=f"input_safe_{feat}"
                        )
                    
                    # Salva permanentemente no session_state para evitar perda de foco ou crash
                    st.session_state.valores_manuais[feat] = valores_usuario[feat]

            if st.button("🚀 Executar Modelo de Precificação Híbrido"):
                df_modelo = df_global[features_selecionadas + [variavel_alvo]].dropna()
                
                if len(df_modelo) < 3:
                    st.error("Amostras insuficientes após remover nulos (mínimo de 3).")
                else:
                    X = df_modelo[features_selecionadas]
                    y = df_modelo[variavel_alvo]

                    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
                    modelo.fit(X, y)
                    r2 = round(modelo.score(X, y), 4)

                    df_alvo = pd.DataFrame([valores_usuario])
                    previsoes = np.array([arvore.predict(df_alvo.values)[0] for arvore in modelo.estimators_])
                    
                    v_medio = float(np.mean(previsoes))
                    v_min = float(np.percentile(previsoes, 15))
                    v_max = float(np.percentile(previsoes, 85))

                    st.success("✅ Modelo treinado com sucesso!")
                    r1, r2_col, r3 = st.columns(3)
                    r1.metric("Valor Mínimo (Segurança)", f"R$ {v_min:,.2f}")
                    r2_col.metric("Valor Estimado (Face)", f"R$ {v_medio:,.2f}")
                    r3.metric("Valor Máximo (Mercado)", f"R$ {v_max:,.2f}")
                    st.caption(f"Acurácia (R²): {r2} | Amostras Saneadas: {len(df_modelo)}")

                    pdf_bytes = gerar_laudo_pdf_ia(
                        tenant_selecionado, variavel_alvo, 
                        {'v_min': v_min, 'v_medio': v_medio, 'v_max': v_max},
                        r2, len(df_modelo),
                        st.session_state.status_juridico_global,
                        st.session_state.score_juridico_global
                    )
                    st.download_button(
                        "📄 Baixar Laudo AVM em PDF",
                        data=pdf_bytes,
                        file_name="laudo_avm_hibrido.pdf",
                        mime="application/pdf",
                    )
        else:
            st.warning("⚠️ Selecione ao menos uma variável independente.")

with aba_juridico:
    st.subheader("📜 Esteira de Risco Jurídico da Matrícula")
    j1, j2 = st.columns(2)
    matricula_ok = j1.checkbox("Matrícula atualizada (menos de 30 dias)", value=True)
    sem_onus = j1.checkbox("Livre de ônus reais (hipoteca, penhora)", value=True)
    sem_acoes = j2.checkbox("Sem ações reipersecutórias", value=True)
    proprietario_ok = j2.checkbox("Vendedor é o proprietário registral", value=True)

    if st.button("⚖️ Processar Análise Jurídica"):
        aprovados = sum([matricula_ok, sem_onus, sem_acoes, proprietario_ok])
        st.session_state.status_juridico_global = aprovados == 4
        st.session_state.score_juridico_global = ["ALTO RISCO", "ALTO RISCO", "RISCO MODERADO", "RISCO BAIXO", "RISCO MÍNIMO"][aprovados]
        if st.session_state.status_juridico_global:
            st.success(f"✅ Documentação APROVADA — {st.session_state.score_juridico_global}")
        else:
            st.error(f"❌ Documentação REPROVADA — {st.session_state.score_juridico_global}")
