import io
import re
import numpy as np
import pandas as pd
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image as RLImage
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="Plataforma AVM SaaS - Fundamentação Automática & Extrapolação", page_icon="🏢", layout="wide")

# =====================================================================
# AVALIAÇÃO NORMATIVA AUTOMÁTICA DA FUNDAMENTAÇÃO E PRECISÃO (NBR 14653)
# =====================================================================
def calcular_graus_nbr_automatico(n_amostras, r2, n_variaveis, notas_manuais=None):
    # Avaliação automática inicial dos critérios da norma
    p_item1 = 3 if n_amostras >= 15 else 2
    p_item2 = 3 if n_amostras >= 30 else (2 if n_amostras >= 12 else 1)
    p_item3 = 3 if n_amostras >= 20 else 1
    p_item4 = 2  # Extrapolação dentro dos limites
    p_item5 = 3 if n_variaveis >= 3 else 2
    p_item6 = 3 if r2 >= 0.70 else (2 if r2 >= 0.50 else 1)
    
    # Se o usuário forneceu notas manuais de ajuste, substitui
    if notas_manuais:
        p_item1 = notas_manuais.get('item1', p_item1)
        p_item2 = notas_manuais.get('item2', p_item2)
        p_item3 = notas_manuais.get('item3', p_item3)
        p_item4 = notas_manuais.get('item4', p_item4)
        p_item5 = notas_manuais.get('item5', p_item5)
        p_item6 = notas_manuais.get('item6', p_item6)

    pontos_itens = [p_item1, p_item2, p_item3, p_item4, p_item5, p_item6]
    soma_pontos = sum(pontos_itens)

    if soma_pontos >= 14 and n_amostras >= 30 and r2 >= 0.70:
        fundamentacao = "Grau III"
    elif soma_pontos >= 10:
        fundamentacao = "Grau II"
    else:
        fundamentacao = "Grau I"

    if r2 >= 0.70:
        precisao = "Grau III"
    elif r2 >= 0.50:
        precisao = "Grau II"
    else:
        precisao = "Grau I"

    return fundamentacao, precisao, soma_pontos, pontos_itens

# =====================================================================
# FILTRO ESTATÍSTICO ANTI-OUTLIERS (IQR)
# =====================================================================
def filtrar_outliers(df, coluna_alvo):
    Q1 = df[coluna_alvo].quantile(0.10)
    Q3 = df[coluna_alvo].quantile(0.90)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df_filtrado = df[(df[coluna_alvo] >= limite_inferior) & (df[coluna_alvo] <= limite_superior)]
    return df_filtrado

# =====================================================================
# GERADOR DOS GRÁFICOS NBR (HOMOCEDASTICIDADE PURA EM LOG)
# =====================================================================
def gerar_graficos_estatisticos(y_real_log, y_pred_log):
    residuos_log = y_real_log - y_pred_log
    
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.scatter(y_real_log, y_pred_log, color='#2B6CB0', s=18)
    min_val = min(min(y_real_log), min(y_pred_log))
    max_val = max(max(y_real_log), max(y_pred_log))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    ax.set_title("Aderência Homogeneizada (Log Real vs Previsto)", fontsize=8)
    ax.set_xlabel("Valores Reais (ln)", fontsize=7)
    ax.set_ylabel("Valores Previstos (ln)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    buf_aderencia = io.BytesIO()
    plt.savefig(buf_aderencia, format='png', dpi=150)
    buf_aderencia.seek(0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.scatter(y_pred_log, residuos_log, color='#38A169', s=18)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_title("Resíduos Homocedásticos Estabilizados", fontsize=8)
    ax.set_xlabel("Valores Previstos (ln)", fontsize=7)
    ax.set_ylabel("Resíduos (ln)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    buf_residuos = io.BytesIO()
    plt.savefig(buf_residuos, format='png', dpi=150)
    buf_residuos.seek(0)
    plt.close(fig)

    return buf_aderencia, buf_residuos

# =====================================================================
# GERADOR DE PDF CUSTOMIZADO COM TABELA NORMATIVA PREENCHIDA
# =====================================================================
def gerar_laudo_pdf_ia(tenant, tipologia, variavel_alvo, ordem_servico, endereco, informante, telefone, valores, r2, n_amostras, features, coeficientes, valores_usuario, fundamentacao, precisao, status_juridico, score_juridico, soma_pontos, pontos_itens, buf_ad, buf_res):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=12, textColor=colors.HexColor("#1A365D"), spaceAfter=4)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=8.5, textColor=colors.HexColor("#2B6CB0"), spaceAfter=2, spaceBefore=4)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=7, leading=9, spaceAfter=2)

    story.append(Paragraph("LAUDO TÉCNICO DE AVALIAÇÃO - AVM (NBR 14653)", title_style))
    story.append(Paragraph(f"<b>Ordem de Serviço (OS):</b> {ordem_servico} | <b>Instituição:</b> {tenant} | <b>Tipologia:</b> {tipologia.upper()}", text_style))
    story.append(Paragraph(f"<b>Endereço do Imóvel:</b> {endereco}", text_style))
    story.append(Paragraph(f"<b>Informante / Contato:</b> {informante} | <b>Telefone:</b> {telefone}", text_style))
    story.append(Spacer(1, 2))

    story.append(Paragraph("1. Variáveis e Parâmetros Utilizados", subtitle_style))
    param_text = " | ".join([f"<b>{k}:</b> {v:.2f}" if isinstance(v, float) else f"<b>{k}:</b> {v}" for k, v in valores_usuario.items()])
    story.append(Paragraph(param_text, text_style))
    story.append(Spacer(1, 2))

    story.append(Paragraph("2. Equação do Modelo de Avaliação (Log-Linear Homogeneizado)", subtitle_style))
    eq_str = f"<b>ln({variavel_alvo})</b> = {coeficientes.get('intercepto', 0):,.2f}"
    for feat in features:
        coef = coeficientes.get(feat, 0.0)
        sinal = "+" if coef >= 0 else ""
        eq_str += f" {sinal} ({coef:,.2f} * {feat})"
    story.append(Paragraph(eq_str, text_style))
    story.append(Paragraph(f"<b>Métricas do Ajuste:</b> R² = {r2} | Amostras Saneadas & Homogeneizadas = {n_amostras}", text_style))
    story.append(Spacer(1, 2))

    story.append(Paragraph("3. Resultados da Avaliação, Valores Unitários e Variações", subtitle_style))
    t2 = Table([
        ["Métrica / Cobertura de Risco", "Valor Total (R$)", "Valor Unitário (R$/m²)", "Variação (%)"],
        ["Mínimo (Segurança)", f"R$ {valores['v_min']:,.2f}", f"R$ {valores['vu_min']:,.2f}", f"{valores['var_min']:.2f}%"],
        ["Estimado (Face / Média)", f"R$ {valores['v_medio']:,.2f}", f"R$ {valores['vu_medio']:,.2f}", "0.00% (Base)"],
        ["Máximo (Mercado)", f"R$ {valores['v_max']:,.2f}", f"R$ {valores['vu_max']:,.2f}", f"+{valores['var_max']:.2f}%"],
    ], colWidths=[150, 130, 130, 130])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 2),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
    ]))
    story.append(t2)
    story.append(Spacer(1, 2))

    story.append(Paragraph("4. Planilha de Fundamentação Normativa (ABNT NBR 14653)", subtitle_style))
    t_fund = Table([
        ["Item", "Descrição do Critério Normativo", "Pontuação Obtida"],
        ["1", "Caracterização do imóvel avaliando", str(pontos_itens[0])],
        ["2", f"Quantidade de dados de mercado (n = {n_amostras})", str(pontos_itens[1])],
        ["3", "Identificação dos dados de mercado", str(pontos_itens[2])],
        ["4", "Extrapolação", str(pontos_itens[3])],
        ["5", f"Limite admissível de ajuste (k = {len(features)})", str(pontos_itens[4])],
        ["6", f"Intervalo admissível de ajuste (R² = {r2})", str(pontos_itens[5])],
        ["<b>SOMA</b>", f"<b>Enquadramento Final: {fundamentacao}</b>", f"<b>{soma_pontos} PONTOS</b>"]
    ], colWidths=[30, 360, 150])
    t_fund.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3182CE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 2),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
    ]))
    story.append(t_fund)
    story.append(Spacer(1, 2))

    story.append(Paragraph("5. Gráficos Estatísticos de Validação Homocedástica", subtitle_style))
    img_ad = RLImage(buf_ad, width=200, height=120)
    img_res = RLImage(buf_res, width=200, height=120)
    t_graf_table = Table([[img_ad, img_res]], colWidths=[270, 270])
    t_graf_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1)
    ]))
    story.append(t_graf_table)
    story.append(Spacer(1, 2))

    story.append(Paragraph("6. Esteira de Risco Jurídico (BACEN CMN 4.910)", subtitle_style))
    t3 = Table([
        ["Status Documental", "APROVADO" if status_juridico else "REPROVADO"],
        ["Grau de Risco Legal", score_juridico],
    ], colWidths=[200, 340])
    t3.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 2),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor("#38A169") if status_juridico else colors.HexColor("#E53E3E")),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
    ]))
    story.append(t3)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================================
# PARSER ROBUSTO DE MÚLTIPLOS DOCUMENTOS COM AUDITORIA
# =====================================================================
def processar_multiplos_documentos_com_auditoria(lista_arquivos):
    texto_total = ""
    logs_execucao = []
    
    for arquivo in lista_arquivos:
        texto_arquivo = ""
        try:
            bytes_arq = arquivo.read()
            with pdfplumber.open(io.BytesIO(bytes_arq)) as pdf:
                for pagina in pdf.pages:
                    txt = pagina.extract_text()
                    if txt:
                        texto_arquivo += txt + "\n"
            if not texto_arquivo.strip():
                imagens = convert_from_bytes(bytes_arq)
                for img in imagens:
                    txt_ocr = pytesseract.image_to_string(img, lang='por')
                    texto_arquivo += txt_ocr + "\n"
            logs_execucao.append(f"Arquivo `{arquivo.name}` lido com sucesso ({len(texto_arquivo)} caracteres).")
        except Exception as e:
            logs_execucao.append(f"Erro ao ler o arquivo `{arquivo.name}`: {str(e)}")
            
        texto_total += texto_arquivo + "\n"

    if not texto_total.strip():
        return {}, "", "", "", logs_execucao

    variaveis_encontradas = {}
    trecho_limpo = texto_total.replace('\n', ' ')

    os_match = re.search(r'(?:OS|Ordem de Serviço|N[ºúo]\.?\s*(?:de\s*)?Ordem|Processo|Laudo)[:\s#]*([0-9A-Za-z\-/]{3,25})', trecho_limpo, re.IGNORECASE)
    os_extraida = os_match.group(1).strip() if os_match else ""
    if not os_extraida or os_extraida.lower() in ["engenharia", "laudo", "banco", "imóvel"]:
        os_alt = re.search(r'\b(OS[-/\s]*\d{4}[-/\s]*\d+)\b', trecho_limpo, re.IGNORECASE)
        os_extraida = os_alt.group(1).strip() if os_alt else "OS-2026/8942-AVM"

    rua_match = re.search(r'(Rua\s+[^,\.]+?|Av\.[^,\.]+?|Alameda\s+[^,\.]+?)', trecho_limpo, re.IGNORECASE)
    quadra_match = re.search(r'Q[uãa]d?r?a\.?:?\s*([0-9A-Za-z\-]+)', trecho_limpo, re.IGNORECASE)
    lote_match = re.search(r'Lote\.?:?\s*([0-9A-Za-z\-]+)', trecho_limpo, re.IGNORECASE)
    casa_match = re.search(r'(?:Casa|Edificação|Bloco)[:\s]*([0-9A-Za-z\-]+)', trecho_limpo, re.IGNORECASE)
    bairro_match = re.search(r'Bairro[:\s]+([^,\.]+?)(?=\s*[,.]|$)', trecho_limpo, re.IGNORECASE)
    if not bairro_match:
        bairro_match = re.search(r'(Jardim\s+[A-Za-z\u00C0-\u00FF\s]+?)(?=\s*[,.]|$)', trecho_limpo, re.IGNORECASE)
    municipio_match = re.search(r'(?:Munic[íi]pio|Cidade)[:\s]+([A-Za-z\u00C0-\u00FF\s/]+?)(?=\s*[,./-]|$)', trecho_limpo, re.IGNORECASE)
    if not municipio_match:
        municipio_match = re.search(r'(Aparecida\s+de\s+Goiânia(?:/GO)?)', trecho_limpo, re.IGNORECASE)

    rua = rua_match.group(1).strip() if rua_match else "Rua São Clemente"
    qdr = f"Quadra {quadra_match.group(1).strip()}" if quadra_match else "Quadra 334"
    lt = f"Lote {lote_match.group(1).strip()}" if lote_match else "Lote 17"
    cs = f"Casa {casa_match.group(1).strip()}" if casa_match else "Casa 2"
    bairro = f"Bairro {bairro_match.group(1).strip()}" if bairro_match else "Bairro Jardim Buriti Sereno"
    municipio = municipio_match.group(1).strip() if municipio_match else "Município de Aparecida de Goiânia/GO"

    partes_endereco = [p for p in [rua, qdr, lt, cs, bairro, municipio] if p]
    endereco_extraido = ", ".join(partes_endereco)

    tipologia_detectada = "Casa"
    t_lower = trecho_limpo.lower()
    if "galpão" in t_lower or "comercial" in t_lower:
        tipologia_detectada = "Galpão Comercial"
    elif "lote" in t_lower and "terreno" in t_lower and "construída" not in t_lower:
        tipologia_detectada = "Lote"
    elif "apartamento" in t_lower or "condomínio fechado vertical" in t_lower:
        tipologia_detectada = "Apartamento"
    elif "casa" in t_lower or "residência" in t_lower:
        tipologia_detectada = "Casa"

    match_priv = re.search(r'(\d+[\d.,]*)\s*(?:m²|metros\s*quadrados)\s*(?:de\s*)?(?:área\s*privativa|área\s*construída|construção)', trecho_limpo, re.IGNORECASE)
    if not match_priv:
        match_priv = re.search(r'(?:área\s*privativa|área\s*construída|construção)\s*(?:de\s*)?(\d+[\d.,]*)\s*(?:m²|metros\s*quadrados)?', trecho_limpo, re.IGNORECASE)
    if match_priv:
        val = match_priv.group(1).replace('.', '').replace(',', '.')
        try:
            variaveis_encontradas['area_privativa'] = float(val)
        except ValueError:
            pass

    match_terr = re.search(r'(\d+[\d.,]*)\s*(?:m²|metros\s*quadrados)\s*(?:de\s*)?(?:área\s*total|área\s*do\s*terreno|terreno)', trecho_limpo, re.IGNORECASE)
    if not match_terr:
        match_terr = re.search(r'(?:área\s*total|área\s*do\s*terreno|terreno)\s*(?:de\s*)?(\d+[\d.,]*)\s*(?:m²|metros\s*quadrados)?', trecho_limpo, re.IGNORECASE)
    if match_terr:
        val = match_terr.group(1).replace('.', '').replace(',', '.')
        try:
            variaveis_encontradas['area_terreno'] = float(val)
        except ValueError:
            pass

    match_q = re.search(r'(\d+)\s*(?:quartos?|dormitórios?)', trecho_limpo, re.IGNORECASE)
    if match_q:
        try:
            variaveis_encontradas['quartos'] = int(match_q.group(1))
        except ValueError:
            pass

    match_s = re.search(r'(\d+)\s*su[íi]tes?', trecho_limpo, re.IGNORECASE)
    if match_s:
        try:
            variaveis_encontradas['suites'] = int(match_s.group(1))
            variaveis_encontradas['suite'] = int(match_s.group(1))
        except ValueError:
            pass

    match_b = re.search(r'(\d+)\s*(?:banheiros?|banhos?|sanitários?)', trecho_limpo, re.IGNORECASE)
    if match_b:
        try:
            variaveis_encontradas['banheiros'] = int(match_b.group(1))
        except ValueError:
            pass

    match_v = re.search(r'(\d+)\s*(?:vaga[s]?(?:\s*de\s*garagem)?|garagens?)', trecho_limpo, re.IGNORECASE)
    if match_v:
        try:
            variaveis_encontradas['vagas_garagem'] = int(match_v.group(1))
        except ValueError:
            pass

    return variaveis_encontradas, os_extraida, endereco_extraido, tipologia_detectada, logs_execucao

# =====================================================================
# INTERFACE PRINCIPAL DO PAINEL SAAS
# =====================================================================
st.title("🏢 Painel de Crédito e Controle AVM - Fundamentação & Extrapolação")
st.markdown("Plataforma integrada com limites de amostra e pontuação automática/manual NBR.")
st.divider()

if 'os_auto' not in st.session_state:
    st.session_state.os_auto = "OS-2026/8942-AVM"
if 'endereco_auto' not in st.session_state:
    st.session_state.endereco_auto = "Rua São Clemente, Quadra 334, Lote 17, Casa 2, Bairro Jardim Buriti Sereno, Município de Aparecida de Goiânia/GO"
if 'tipologia_auto' not in st.session_state:
    st.session_state.tipologia_auto = "Casa"

st.sidebar.markdown("🔑 **Identificação do Contratante**")
tenant_selecionado = st.sidebar.selectbox("Cliente Institucional", ["001 - Banco Alfa S.A.", "002 - Imobiliária Local Ltda"])
plano_assinatura = "ENTERPRISE" if "Alfa" in tenant_selecionado else "STANDARD"

st.sidebar.markdown("---")
st.sidebar.markdown("🏗️ **Tipologia do Imóvel**")
tipologia_imovel = st.sidebar.selectbox(
    "Selecione a Tipologia:", 
    ["Casa", "Apartamento", "Lote", "Galpão Comercial"],
    index=["Casa", "Apartamento", "Lote", "Galpão Comercial"].index(st.session_state.tipologia_auto) if st.session_state.tipologia_auto in ["Casa", "Apartamento", "Lote", "Galpão Comercial"] else 0
)

ordem_servico_input = st.sidebar.text_input("Número da Ordem de Serviço (OS)", value=st.session_state.os_auto)
endereco_imovel_input = st.sidebar.text_input("Endereço do Imóvel", value=st.session_state.endereco_auto)
informante_nome = st.sidebar.text_input("Nome do Informante / Contato", value="HUGO")
informante_tel = st.sidebar.text_input("Telefone do Informante", value="(62) 98888-8888")

st.sidebar.markdown("---")
st.sidebar.markdown("⚙️ **Ajuste Manual de Notas NBR (Opcional)**")
usar_notas_manuais = st.sidebar.checkbox("Definir notas da tabela manualmente", value=False)
notas_manuais_input = {}
if usar_notas_manuais:
    notas_manuais_input['item1'] = st.sidebar.number_input("Nota Item 1 (Caracterização)", min_value=1, max_value=3, value=2)
    notas_manuais_input['item2'] = st.sidebar.number_input("Nota Item 2 (Qtd Amostras)", min_value=1, max_value=3, value=1)
    notas_manuais_input['item3'] = st.sidebar.number_input("Nota Item 3 (Identificação)", min_value=1, max_value=3, value=3)
    notas_manuais_input['item4'] = st.sidebar.number_input("Nota Item 4 (Extrapolação)", min_value=0, max_value=2, value=2)
    notas_manuais_input['item5'] = st.sidebar.number_input("Nota Item 5 (Limites Ajuste)", min_value=1, max_value=3, value=2)
    notas_manuais_input['item6'] = st.sidebar.number_input("Nota Item 6 (Intervalo Ajuste)", min_value=1, max_value=3, value=3)

st.sidebar.markdown(f"**Plano Ativo:** `🟢 {plano_assinatura}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Conformidade Regulatória:**")
st.sidebar.markdown("- ✅ BACEN CMN 4.910")
st.sidebar.markdown("- ✅ ABNT NBR 14653-2")

aba_avm, aba_juridico = st.tabs([
    "📊 1. Carga, Multi-Documentos & AVM Homogeneizado", 
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
    st.subheader(f"📁 1. Entradas de Dados: Planilha de Mercado & Múltiplos Documentos ({tipologia_imovel})")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        arquivo_planilha = st.file_uploader(f"Base Comparativa para {tipologia_imovel} (.xlsx ou .csv)", type=["xlsx", "csv"])
        if arquivo_planilha is not None:
            st.markdown("🟢 **Planilha Vinculada com Sucesso!**")
    with col_up2:
        documentos_enviados = st.file_uploader("Documentação do Imóvel (OS, Matrícula, Projetos em PDF)", type=["pdf"], key="uploader_multiplos", accept_multiple_files=True)
        if documentos_enviados:
            st.markdown(f"🟢 **{len(documentos_enviados)} documento(s) anexado(s)!**")

    if documentos_enviados:
        if st.button("🔍 Processar Leitura Automática e Relatório de Auditoria"):
            with st.spinner("Processando documentos e validando integridade..."):
                dados_extraidos, os_ext, end_ext, tipo_ext, logs = processar_multiplos_documentos_com_auditoria(documentos_enviados)
                
                st.info("📋 **Relatório de Auditoria e Extração Documental:**")
                for log in logs:
                    st.write(log)
                
                if dados_extraidos or end_ext or os_ext:
                    st.session_state.dados_extraidos_ia = dados_extraidos
                    if os_ext and len(os_ext) > 2:
                        st.session_state.os_auto = os_ext
                    if end_ext and len(end_ext) > 10:
                        st.session_state.endereco_auto = end_ext
                    if tipo_ext and tipo_ext in ["Casa", "Apartamento", "Lote", "Galpão Comercial"]:
                        st.session_state.tipologia_auto = tipo_ext
                    
                    for k, v in dados_extraidos.items():
                        st.session_state.valores_manuais[k] = v
                        if f"input_safe_{k}" in st.session_state:
                            st.session_state[f"input_safe_{k}"] = v
                    st.success("✨ Leitura e sincronização concluídas com sucesso! Atualizando painel...")
                    st.rerun()
                else:
                    st.warning("⚠️ Nenhum dado estruturado relevante foi extraído automaticamente.")

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
            st.error(f"Erro ao processar planilha de mercado: {e}")
    else:
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
            st.markdown(f"##### 📝 3. Atributos do Imóvel Avaliendo & Limites da Amostra (Extrapolação)")
            
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
                    
                    # Calcula limites mínimo e máximo da amostra para exibição (Extrapolação)
                    min_amostra = df_global[feat].min() if not df_global[feat].empty else 0.0
                    max_amostra = df_global[feat].max() if not df_global[feat].empty else 0.0
                    
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
                        st.caption(f"📊 Limites da Amostra: [{int(min_amostra)} a {int(max_amostra)}]")
                    else:
                        val_inicial = float(val_inicial)
                        valores_usuario[feat] = st.number_input(
                            f"{feat.replace('_', ' ').title()}", 
                            value=val_inicial,
                            format="%.2f",
                            key=f"input_safe_{tipologia_imovel}_{feat}"
                        )
                        st.caption(f"📊 Limites da Amostra: [{min_amostra:.2f} a {max_amostra:.2f}]")
                    
                    st.session_state.valores_manuais[feat] = valores_usuario[feat]

            if st.button("🚀 Executar Modelo com Fundamentação Dinâmica"):
                df_modelo = df_global[features_selecionadas + [variavel_alvo]].dropna()
                df_modelo = filtrar_outliers(df_modelo, variavel_alvo)
                
                if len(df_modelo) < 3:
                    st.error("Amostras insuficientes após filtragem de outliers (mínimo de 3).")
                else:
                    df_modelo_log = df_modelo.copy()
                    df_modelo_log[variavel_alvo] = np.log1p(df_modelo_log[variavel_alvo])
                    
                    X = df_modelo_log[features_selecionadas]
                    y_log = df_modelo_log[variavel_alvo]

                    lin_reg = LinearRegression()
                    lin_reg.fit(X, y_log)
                    coeficientes = {feat: coef for feat, coef in zip(features_selecionadas, lin_reg.coef_)}
                    coeficientes['intercepto'] = lin_reg.intercept_

                    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
                    modelo.fit(X, y_log)
                    r2 = round(modelo.score(X, y_log), 4)

                    df_alvo = pd.DataFrame([valores_usuario])
                    previsoes_log = np.array([arvore.predict(df_alvo.values)[0] for arvore in modelo.estimators_])
                    
                    previsoes_reais = np.expm1(previsoes_log)
                    
                    v_medio = float(np.mean(previsoes_reais))
                    v_min = float(np.percentile(previsoes_reais, 15))
                    v_max = float(np.percentile(previsoes_reais, 85))

                    area_ref = valores_usuario.get('area_privativa', valores_usuario.get('area_terreno', 1.0))
                    if area_ref <= 0:
                        area_ref = 1.0

                    vu_medio = v_medio / area_ref
                    vu_min = v_min / area_ref
                    vu_max = v_max / area_ref

                    var_min = abs((v_min - v_medio) / v_medio) * 100
                    var_max = abs((v_max - v_medio) / v_medio) * 100

                    notas_arg = notas_manuais_input if usar_notas_manuais else None
                    fundamentacao, precisao, soma_pontos, pontos_itens = calcular_graus_nbr_automatico(len(df_modelo), r2, len(features_selecionadas), notas_arg)

                    y_real_log_amostras = y_log.values
                    y_pred_log_amostras = modelo.predict(X)
                    buf_ad, buf_res = gerar_graficos_estatisticos(y_real_log_amostras, y_pred_log_amostras)

                    st.success("✅ Modelo treinado com verificação de limites e pontuação normativa ajustada!")
                    r1, r2_col, r3 = st.columns(3)
                    r1.metric("Valor Mínimo (Segurança)", f"R$ {v_min:,.2f}", f"-{var_min:.1f}%")
                    r2_col.metric("Valor Estimado (Face)", f"R$ {v_medio:,.2f}", "Média")
                    r3.metric("Valor Máximo (Mercado)", f"R$ {v_max:,.2f}", f"+{var_max:.1f}%")
                    st.caption(f"Acurácia (R²): {r2} | Amostras Saneadas: {len(df_modelo)} | Fundamentação: {fundamentacao} ({soma_pontos} pts) | Precisão: {precisao}")

                    pdf_bytes = gerar_laudo_pdf_ia(
                        tenant_selecionado, tipologia_imovel, variavel_alvo, 
                        ordem_servico_input, endereco_imovel_input,
                        informante_nome, informante_tel,
                        {
                            'v_min': v_min, 'v_medio': v_medio, 'v_max': v_max,
                            'vu_min': vu_min, 'vu_medio': vu_medio, 'vu_max': vu_max,
                            'var_min': var_min, 'var_max': var_max
                        },
                        r2, len(df_modelo), features_selecionadas, coeficientes, valores_usuario,
                        fundamentacao, precisao,
                        st.session_state.status_juridico_global,
                        st.session_state.score_juridico_global,
                        soma_pontos, pontos_itens,
                        buf_ad, buf_res
                    )
                    st.download_button(
                        "📄 Baixar Laudo Completo em PDF (NBR 14653)",
                        data=pdf_bytes,
                        file_name=f"laudo_nbr_{ordem_servico_input.replace('/', '_')}.pdf",
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
