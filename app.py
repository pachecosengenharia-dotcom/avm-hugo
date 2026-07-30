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

st.set_page_config(page_title="Plataforma AVM SaaS - Multi-Tipologia", page_icon="🏢", layout="wide")

# =====================================================================
# GERADOR DE PDF CUSTOMIZADO
# =====================================================================
def gerar_laudo_pdf_ia(tenant, tipologia, variavel_alvo, valores, r2, n_amostras, status_juridico, score_juridico):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#1A365D"), spaceAfter=15)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#2B6CB0"), spaceAfter=8)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=6)

    story.append(Paragraph(f"LAUDO CORE AVM - INTELIGÊNCIA ARTIFICIAL", title_style))
    story.append(Paragraph(f"<b>Instituição Solicitante:</b> {tenant}", text_style))
    story.append(Paragraph(f"<b>Tipologia do Imóvel:</b> {tipologia.upper()}", text_style))
    story.append(Paragraph(f"<b>Variável Alvo Precificada:</b> {variavel_alvo.upper()}", text_style))
    story.append(Paragraph("<b>Metodologia Core:</b> Random Forest Regressor | NBR 14653-2", text_style))
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

    # Área Privativa / Construída
    match_privativa = re.search(r'([\d.,]+)\s*metros\s*quadrados\s*de\s*área\s*privativa', trecho_limpo, re.IGNORECASE)
    if not match_privativa:
        match_privativa = re.search(r'área\s*(?:privativa|construída)\s*(?:de\s*)?([\d.,]+)', trecho_limpo, re.IGNORECASE)
    if match_privativa:
        val = match_privativa.group(1).replace('.', '').replace(',', '.').replace('!', '1')
        try:
            variaveis_encontradas['area_privativa'] = float(val)
        except ValueError:
            pass

    # Área do Terreno
    match_terreno = re.search(r'com\s*área\s*total\s*de\s*([\d.,]+)\s*metros\s*quadrados.*?fração', trecho_limpo, re.IGNORECASE)
    if not match_terreno:
        match_terreno = re.search(r'área\s*(?:total|do\s*terreno)\s*(?:de\s*)?([\d.,]+)', trecho_limpo, re.IGNORECASE)
    if match_terreno:
        val = match_terreno.group(1).replace('.', '').replace(',', '.').replace('!', '1')
        try:
            variaveis_encontradas['area_terreno'] = float(val)
        except ValueError:
            pass

    match_divisao = re.search(r'divisão\s*interna[:\s]*(.*?)(?:edificada|lote|$)', trecho_limpo, re.IGNORECASE)
    trecho_divisao = match_divisao.group(1) if match_divisao else trecho_limpo

    # Quartos
    match_quartos = re.search(r'(\d+)\s*\([^)]+\)\s*quartos', trecho_divisao, re.IGNORECASE)
    if not match_quartos:
        match_quartos = re.search(r'(\d+)\s*quarto[s]?', trecho_divisao, re.IGNORECASE)
    if match_quartos:
        try:
            variaveis_encontradas['quartos'] = int(match_quartos.group(1))
        except ValueError:
            pass

    # Suítes
    match_suites = re.search(r'sendo\s*(?:0?(\d+)|um|dois)', trecho_divisao, re.IGNORECASE)
    val_suites = 1 
    if match_suites and match_suites.group(1):
        try:
            val_suites = int(match_suites.group(1))
        except ValueError:
            pass
    elif "sendo um" in trecho_divisao.lower() or "sendo 01" in trecho_divisao.lower():
        val_suites = 1
    variaveis_encontradas['suites'] = val_suites
    variaveis_encontradas['suite'] = val_suites

    # Banheiros
    match_banheiros = re.search(r'(\d+)\s*\([^)]+\)\s*banho', trecho_divisao, re.IGNORECASE)
    if not match_banheiros:
        match_banheiros = re.search(r'(\d+)\s*banho', trecho_divisao, re.IGNORECASE)
    if match_banheiros:
        try:
            variaveis_encontradas['banheiros'] = int(match_banheiros.group(1))
        except ValueError:
            pass
    else:
        variaveis_encontradas['banheiros'] = 1

    # Vagas
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
# INTERFACE PRINCIPAL DO PAINEL SAAS (SIDEBAR IDÊNTICA AO SOLICITADO)
# =====================================================================
st.title("🏢 Painel de Crédito e Controle AVM - Multi-Tipologia")
st.markdown("Plataforma agnóstica para Modelagem Automatizada de Imóveis (Residencial e Comercial).")
st.divider()

st.sidebar.markdown("🔑 **Identificação do Contratante**")
tenant_selecionado = st.sidebar.selectbox("Cliente Institucional", ["001 - Banco Alfa S.A.", "002 - Imobiliária Local Ltda"])
plano_assinatura = "ENTERPRISE" if "Alfa" in tenant_selecionado else "STANDARD"

st.sidebar.markdown(f"**Plano Ativo:** `🟢 {plano_assinatura}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Conformidade Regulatória:**")
st.sidebar.markdown("- ✅ BACEN CMN 4.910")
st.sidebar.markdown("- ✅ ABNT NBR 14653-2")

# SELETOR DE TIPOLOGIA IMOBILIÁRIA (4 Tipologias)
st.sidebar.markdown("---")
st.sidebar.markdown("🏗️ **Tipologia do Imóvel**")
tipologia_imovel = st.sidebar.selectbox(
    "Selecione a Tipologia:", 
    ["Apartamento", "Casa", "Lote", "Galpão Comercial"]
)

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
    st.subheader(f"📁 1. Entradas de Dados: Planilha de Mercado & Certidão ({tipologia_imovel})")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        arquivo_planilha = st.file_uploader(f"Base Comparativa para {tipologia_imovel} (.xlsx ou .csv)", type=["xlsx", "csv"])
        if arquivo_planilha is not None:
            st.markdown("🟢 **Planilha Vinculada com Sucesso!**")
    with col_up2:
        documento_enviado = st.file_uploader("Certidão de Ônus / Matrícula em PDF", type=["pdf"])
        if documento_enviado is not None:
            st.markdown("🟢 **Certidão de Ônus Anexada!**")

    if documento_enviado is not None:
        dados_extraidos, _ = extrair_variaveis_de_documento(documento_enviado)
        if dados_extraidos:
            st.session_state.dados_extraidos_ia = dados_extraidos
            for k, v in dados_extraidos.items():
                st.session_state.valores_manuais[k] = v
                if f"input_safe_{k}" in st.session_state:
                    st.session_state[f"input_safe_{k}"] = v
            st.success(f"✨ Certidão lida e sincronizada com sucesso para {tipologia_imovel}!")

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
            st.success(f"✅ Base de mercado processada! {len(df_global)} amostras carregadas.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    else:
        # Base demonstrativa adaptada à tipologia selecionada
        if tipologia_imovel == "Lote":
            data_padrao = {
                'valor_total_declarado': [200000, 220000, 250000, 300000, 350000, 180000],
                'area_terreno': [300.0, 350.0, 400.0, 450.0, 500.0, 250.0],
                'indice_fiscal': [1000.0, 1100.0, 1200.0, 1500.0, 1600.0, 900.0],
                'estado_de_conservacao': [3, 4, 3, 5, 4, 3],
                'padrao_de_acabamento': [2, 2, 3, 3, 3, 2],
                'idade_aparente': [0, 0, 0, 0, 0, 0],
                'evento': [1, 1, 2, 1, 2, 1],
                'data_do_evento': [2024, 2024, 2025, 2025, 2026, 2026]
            }
        elif tipologia_imovel == "Galpão Comercial":
            data_padrao = {
                'valor_total_declarado': [1200000, 1500000, 1800000, 2200000, 2500000, 1000000],
                'area_privativa': [600.0, 750.0, 900.0, 1100.0, 1300.0, 500.0],
                'area_terreno': [1000.0, 1200.0, 1500.0, 1800.0, 2000.0, 800.0],
                'pe_direito': [6.0, 7.0, 8.0, 8.5, 9.0, 6.0],
                'vagas_garagem': [5, 8, 10, 12, 15, 4],
                'indice_fiscal': [3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 3000.0],
                'estado_de_conservacao': [4, 4, 3, 5, 4, 3],
                'padrao_de_acabamento': [3, 3, 4, 4, 5, 2],
                'idade_aparente': [5, 8, 3, 12, 6, 10],
                'evento': [1, 1, 2, 1, 2, 1],
                'data_do_evento': [2024, 2024, 2025, 2025, 2026, 2026]
            }
        else: # Apartamento ou Casa
            data_padrao = {
                'valor_total_declarado': [450000, 480000, 510000, 750000, 820000, 350000],
                'area_privativa': [75.0, 78.0, 80.0, 85.0, 92.0, 60.0],
                'area_terreno': [200.0, 220.0, 250.0, 360.0, 400.0, 0.0],
                'quartos': [2, 2, 3, 3, 3, 1],
                'suites': [1, 1, 1, 2, 2, 0],
                'banheiros': [1, 1, 2, 2, 2, 1],
                'vagas_garagem': [1, 2, 2, 2, 3, 1],
                'indice_fiscal': [1200.0, 1250.0, 1300.0, 3200.0, 3300.0, 1500.0],
                'estado_de_conservacao': [3, 4, 3, 5, 4, 3],
                'padrao_de_acabamento': [2, 3, 2, 4, 3, 2],
                'idade_aparente': [5, 10, 2, 15, 8, 3],
                'evento': [1, 1, 2, 1, 2, 1],
                'data_do_evento': [2024, 2024, 2025, 2025, 2026, 2026]
            }
            
        df_global = pd.DataFrame(data_padrao)
        st.info(f"ℹ️ Utilizando base padrão demonstrativa para a tipologia: **{tipologia_imovel}**.")

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
            st.markdown(f"##### 📝 3. Atributos do Imóvel Avaliendo ({tipologia_imovel})")
            
            dados_ia = st.session_state.get('dados_extraidos_ia', {})
            campos_inteiros = [
                'quartos', 'suites', 'suite', 'banheiros', 'vagas', 'vagas_garagem', 'garagem',
                'estado_de_conservacao', 'conservacao', 'padrao_de_acabamento', 'acabamento', 
                'idade_aparente', 'idade', 'evento', 'data_do_evento', 'ano', 'pe_direito'
            ]
            
            valores_usuario = {}
            cols_inputs = st.columns(len(features_selecionadas))
            
            for i, feat in enumerate(features_selecionadas):
                with cols_inputs[i % len(cols_inputs)]:
                    eh_inteiro = any(ci in feat.lower() for ci in campos_inteiros)
                    
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
                            key=f"input_safe_{tipologia_imovel}_{feat}"
                        )
                    else:
                        val_inicial = float(val_inicial)
                        valores_usuario[feat] = st.number_input(
                            f"{feat.replace('_', ' ').title()}", 
                            value=val_inicial,
                            format="%.2f",
                            key=f"input_safe_{tipologia_imovel}_{feat}"
                        )
                    
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
                        tenant_selecionado, tipologia_imovel, variavel_alvo, 
                        {'v_min': v_min, 'v_medio': v_medio, 'v_max': v_max},
                        r2, len(df_modelo),
                        st.session_state.status_juridico_global,
                        st.session_state.score_juridico_global
                    )
                    st.download_button(
                        "📄 Baixar Laudo AVM em PDF",
                        data=pdf_bytes,
                        file_name=f"laudo_avm_{tipologia_imovel.lower().replace(' ', '_')}.pdf",
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
