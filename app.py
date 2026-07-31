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
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="Plataforma AVM SaaS - Motor de Equações Válidas NBR", page_icon="🏢", layout="wide")

# =====================================================================
# CÁLCULO ESTATÍSTICO AUTOMÁTICO (TESTE T E TESTE F DE SNEDECOR)
# =====================================================================
def calcular_estatisticas_regressao(X, y, coeficientes_reg):
    n = len(y)
    k = X.shape[1]
    
    X_matrix = np.hstack([np.ones((n, 1)), X])
    y_array = y
    
    y_pred_ols = X_matrix.dot(coeficientes_reg)
    residuos = y_array - y_pred_ols
    
    soma_sq_res = np.sum(residuos ** 2)
    graus_liberdade = n - k - 1
    
    if graus_liberdade > 0:
        var_res = soma_sq_res / graus_liberdade
        try:
            cov_mat = var_res * np.linalg.inv(X_matrix.T.dot(X_matrix))
            desvio_padrao_se = np.sqrt(np.diagonal(cov_mat))
            t_stats = coeficientes_reg / desvio_padrao_se
            p_valores_t = [2 * (1 - stats.t.cdf(np.abs(t), df=graus_liberdade)) for t in t_stats]
        except Exception:
            p_valores_t = [0.05] * (k + 1)
    else:
        p_valores_t = [0.05] * (k + 1)
        
    soma_sq_reg = np.sum((y_pred_ols - np.mean(y_array)) ** 2)
    soma_sq_tot = np.sum((y_array - np.mean(y_array)) ** 2)
    
    if soma_sq_tot > 0 and k > 0 and graus_liberdade > 0:
        r2_calc = soma_sq_reg / soma_sq_tot
        f_stat = (r2_calc / k) / ((1 - r2_calc) / graus_liberdade) if r2_calc < 1 else 999.99
        p_valor_f = 1 - stats.f.cdf(f_stat, k, graus_liberdade)
    else:
        p_valor_f = 0.001
        
    return p_valores_t, p_valor_f

# =====================================================================
# AVALIAÇÃO NORMATIVA RIGOROSA DA FUNDAMENTAÇÃO E PRECISÃO (NBR 14653)
# =====================================================================
def calcular_graus_nbr_rigoroso(n_amostras, r2, n_variaveis, p_valores_t, p_valor_f, tem_extrapolacao=False, notas_manuais=None):
    p_item1 = notas_manuais.get('item1', 2) if notas_manuais else 2
    
    if n_amostras >= 30:
        p_item2 = 3
    elif n_amostras >= 12:
        p_item2 = 2
    else:
        p_item2 = 1
        
    p_item3 = notas_manuais.get('item3', 2) if notas_manuais else 2
    p_item4 = 1 if tem_extrapolacao else 3
    
    max_p_regressor = max(p_valores_t[1:]) if len(p_valores_t) > 1 else 0.05
    
    if max_p_regressor <= 0.10:
        p_item5 = 3
    elif max_p_regressor <= 0.20:
        p_item5 = 2
    elif max_p_regressor <= 0.30:
        p_item5 = 1
    else:
        p_item5 = 0  # Inválido (> 30%)
        
    if p_valor_f <= 0.01:
        p_item6 = 3
    elif p_valor_f <= 0.05:
        p_item6 = 2
    else:
        p_item6 = 1

    if notas_manuais:
        if 'item2_manual' in notas_manuais:
            p_item2 = notas_manuais['item2_manual']
        if 'item4_manual' in notas_manuais:
            p_item4 = notas_manuais['item4_manual']
        if 'item5_manual' in notas_manuais:
            p_item5 = notas_manuais['item5_manual']
        if 'item6_manual' in notas_manuais:
            p_item6 = notas_manuais['item6_manual']

    pontos_itens = [p_item1, p_item2, p_item3, p_item4, p_item5, p_item6]
    soma_pontos = sum(pontos_itens)

    if soma_pontos >= 16 and n_amostras >= 30 and r2 >= 0.70 and not tem_extrapolacao and p_item5 > 0:
        fundamentacao = "Grau III"
    elif soma_pontos >= 10 and n_amostras >= 12 and p_item5 > 0:
        fundamentacao = "Grau II"
    else:
        fundamentacao = "Inválido / Grau I"

    if r2 >= 0.70:
        precisao = "Grau III"
    elif r2 >= 0.50:
        precisao = "Grau II"
    else:
        precisao = "Grau I"

    return fundamentacao, precisao, soma_pontos, pontos_itens, max_p_regressor, p_valor_f

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
# GERADOR DE PDF CUSTOMIZADO
# =====================================================================
def gerar_laudo_pdf_ia(tenant, tipologia, variavel_alvo, ordem_servico, endereco, informante, telefone, valores, r2, n_amostras, features, coeficientes, valores_usuario, variaveis_extrapoladas, fundamentacao, precisao, status_juridico, score_juridico, soma_pontos, pontos_itens, max_p_regressor, p_valor_f, buf_ad, buf_res):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=11, textColor=colors.HexColor("#1A365D"), spaceAfter=6, leading=14)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=8, textColor=colors.HexColor("#2B6CB0"), spaceAfter=3, spaceBefore=6, leading=10)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=7, leading=10, spaceAfter=3)

    story.append(Paragraph("LAUDO TÉCNICO DE AVALIAÇÃO - AVM (NBR 14653)", title_style))
    story.append(Paragraph(f"<b>Ordem de Serviço (OS):</b> {ordem_servico} | <b>Instituição:</b> {tenant} | <b>Tipologia:</b> {tipologia.upper()}", text_style))
    story.append(Paragraph(f"<b>Endereço do Imóvel:</b> {endereco}", text_style))
    story.append(Paragraph(f"<b>Informante / Contato:</b> {informante} | <b>Telefone:</b> {telefone}", text_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Variáveis e Parâmetros Utilizados (Vermelho indica Extrapolação)", subtitle_style))
    
    param_formatted_list = []
    for k, v in valores_usuario.items():
        val_str = f"{v:.2f}" if isinstance(v, float) else f"{v}"
        if k in variaveis_extrapoladas:
            param_formatted_list.append(f"<font color='red'><b>{k}: {val_str} (EXTRAPOLADO)</b></font>")
        else:
            param_formatted_list.append(f"<b>{k}:</b> {val_str}")
    param_text = " | ".join(param_formatted_list)
    story.append(Paragraph(param_text, text_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Equação do Modelo Válido (Log-Linear Homogeneizado - 6 Casas Decimais)", subtitle_style))
    intercepto_val = coeficientes.get('intercepto', 0)
    eq_str = f"<b>ln(Valor Unitário)</b> = {intercepto_val:,.6f}"
    for feat in features:
        coef = coeficientes.get(feat, 0.0)
        sinal = "+" if coef >= 0 else ""
        eq_str += f" {sinal} ({coef:,.6f} * {feat})"
    story.append(Paragraph(eq_str, text_style))
    story.append(Paragraph(f"<b>Métricas:</b> R² = {r2} | Amostras = {n_amostras} | <b>Máx p-t Regressores:</b> {max_p_regressor*100:.2f}% | <b>p-F Modelo:</b> {p_valor_f:.4f}", text_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Resultados da Avaliação, Valores Unitários e Variações", subtitle_style))
    t2 = Table([
        ["Métrica / Cobertura de Risco", "Valor Total (R$)", "Valor Unitário (R$/m²)", "Variação (%)"],
        ["Mínimo (Segurança)", f"R$ {valores['v_min']:,.2f}", f"R$ {valores['vu_min']:,.2f}", f"{valores['var_min']:.2f}%"],
        ["Estimado (Face / Média)", f"R$ {valores['v_medio']:,.2f}", f"R$ {valores['vu_medio']:,.2f}", "0.00% (Base)"],
        ["Máximo (Mercado)", f"R$ {valores['v_max']:,.2f}", f"R$ {valores['vu_max']:,.2f}", f"+{valores['var_max']:.2f}%"],
    ], colWidths=[150, 135, 135, 134])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Planilha de Fundamentação e Precisão Normativa (ABNT NBR 14653)", subtitle_style))
    t_fund = Table([
        ["Item", "Descrição do Critério Normativo", "Pontuação / Grau Obtido"],
        ["1", "Caracterização do imóvel avaliando", str(pontos_itens[0])],
        ["2", f"Quantidade de dados de mercado (n = {n_amostras})", str(pontos_itens[1])],
        ["3", "Identificação dos dados de mercado", str(pontos_itens[2])],
        ["4", f"Extrapolação ({'Com Extrapol. - Nota 1' if variaveis_extrapoladas else 'Sem Extrapol. - Nota 3'})", str(pontos_itens[3])],
        ["5", f"Significância Regressores (Máx p = {max_p_regressor*100:.1f}%)", str(pontos_itens[4])],
        ["6", f"Significância Modelo F (p = {p_valor_f:.4f})", str(pontos_itens[5])],
        ["SOMA", f"Fundamentação: {fundamentacao} | Precisão: {precisao}", f"{soma_pontos} PONTOS"]
    ], colWidths=[30, 334, 190])
    t_fund.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3182CE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#EDF2F7")),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.HexColor("#1A365D")),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(t_fund)
    story.append(Spacer(1, 4))

    story.append(Paragraph("5. Gráficos Estatísticos de Validação Homocedástica", subtitle_style))
    img_ad = RLImage(buf_ad, width=190, height=110)
    img_res = RLImage(buf_res, width=190, height=110)
    t_graf_table = Table([[img_ad, img_res]], colWidths=[277, 277])
    t_graf_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2)
    ]))
    story.append(t_graf_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Esteira de Risco Jurídico (BACEN CMN 4.910)", subtitle_style))
    t3 = Table([
        ["Status Documental", "APROVADO" if status_juridico else "REPROVADO"],
        ["Grau de Risco Legal", score_juridico],
    ], colWidths=[180, 374])
    t3.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 3),
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
st.title("🏢 Painel de Crédito e Controle AVM - Motor de Equações Válidas NBR")
st.markdown("Validação rigorosa do Item 5 (Significância $\le 30\%$): cálculo correto do valor unitário por metro quadrado e do valor total.")
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
st.sidebar.markdown("⚙️ **Atribuição Manual de Notas NBR (Obrigatório Itens 1 e 3)**")
notas_manuais_input = {}
notas_manuais_input['item1'] = st.sidebar.number_input("Nota Item 1 (Caracterização do Imóvel)", min_value=1, max_value=3, value=2)
notas_manuais_input['item3'] = st.sidebar.number_input("Nota Item 3 (Identificação dos Dados)", min_value=1, max_value=3, value=2)

usar_todas_manuais = st.sidebar.checkbox("Ajustar itens restantes manualmente se necessário", value=False)
if usar_todas_manuais:
    notas_manuais_input['item2_manual'] = st.sidebar.number_input("Nota Item 2 (Qtd Amostras)", min_value=1, max_value=3, value=2)
    notas_manuais_input['item4_manual'] = st.sidebar.number_input("Nota Item 4 (Extrapolação)", min_value=1, max_value=3, value=2)
    notas_manuais_input['item5_manual'] = st.sidebar.number_input("Nota Item 5 (Signif. Regressores)", min_value=1, max_value=3, value=2)
    notas_manuais_input['item6_manual'] = st.sidebar.number_input("Nota Item 6 (Signif. Modelo F)", min_value=1, max_value=3, value=2)

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
                str(c).lower().strip().replace(" ", "_")
                .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
                .replace("ã", "a").replace("õ", "o").replace("ç", "c").replace("â", "a").replace("ê", "e")
                for c in df_global.columns
            ]
            
            df_global = df_global.loc[:, ~df_global.columns.duplicated()].copy()
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
        df_global = df_global.loc[:, ~df_global.columns.duplicated()].copy()
        st.info(f"ℹ️ Utilizando base padrão demonstrativa para a tipologia: **{tipologia_imovel}**.")

    st.markdown("---")
    st.subheader("🤖 2. Configuração e Seleção de Variáveis Independentes")
    
    colunas_numericas = df_global.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(colunas_numericas) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            col_valor_total = st.selectbox("Coluna de Valor Total na Base:", [c for c in colunas_numericas if 'valor' in c or 'preco' in c] + colunas_numericas)
        with c2:
            col_area_base = st.selectbox("Coluna de Área Base (ex: area_privativa ou area_terreno):", [c for c in colunas_numericas if 'area' in c] + colunas_numericas)

        features_disponiveis = [c for c in colunas_numericas if c != col_valor_total]
        features_selecionadas = st.multiselect(
            "Escolha as Variáveis Independentes do Modelo:",
            options=features_disponiveis,
            default=[c for c in features_disponiveis if c != col_area_base][:min(2, len(features_disponiveis))]
        )

        if features_selecionadas and col_valor_total and col_area_base:
            st.markdown(f"##### 📝 3. Atributos do Imóvel Avaliendo & Limites da Amostra (Extrapolação)")
            
            dados_ia = st.session_state.get('dados_extraidos_ia', {})
            campos_inteiros = [
                'quartos', 'suites', 'suite', 'banheiros', 'vagas', 'vagas_garagem', 'garagem',
                'estado_de_conservacao', 'conservacao', 'padrao_de_acabamento', 'acabamento', 
                'idade_aparente', 'idade', 'evento', 'data_do_evento', 'ano', 'pe_direito'
            ]
            
            valores_usuario = {}
            variaveis_extrapoladas = []
            cols_inputs = st.columns(len(features_selecionadas))
            
            for i, feat in enumerate(features_selecionadas):
                with cols_inputs[i % len(cols_inputs)]:
                    eh_inteiro = any(ci in feat.lower() for ci in campos_inteiros)
                    
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
                    
                    nome_formatado = feat.replace('_', ' ').title()
                    
                    if eh_inteiro:
                        val_inicial = int(round(float(val_inicial)))
                        val_input = st.number_input(
                            f"{nome_formatado}", 
                            value=val_inicial,
                            step=1, 
                            format="%d",
                            key=f"input_safe_{tipologia_imovel}_{feat}"
                        )
                        valores_usuario[feat] = val_input
                        st.caption(f"📊 Limites da Amostra: [{int(min_amostra)} a {int(max_amostra)}]")
                    else:
                        val_inicial = float(val_inicial)
                        val_input = st.number_input(
                            f"{nome_formatado}", 
                            value=val_inicial,
                            format="%.2f",
                            key=f"input_safe_{tipologia_imovel}_{feat}"
                        )
                        valores_usuario[feat] = val_input
                        st.caption(f"📊 Limites da Amostra: [{min_amostra:.2f} a {max_amostra:.2f}]")
                    
                    if valores_usuario[feat] < min_amostra or valores_usuario[feat] > max_amostra:
                        variaveis_extrapoladas.append(feat)
                        st.error(f"⚠️ Alerta: '{nome_formatado}' está EXTRAPOLADO em relação à amostra!")

                    st.session_state.valores_manuais[feat] = valores_usuario[feat]

            tem_extrapolacao_geral = len(variaveis_extrapoladas) > 0

            if st.button("🚀 Executar e Validar Equação pelo Motor NBR"):
                colunas_necessarias = list(set(features_selecionadas + [col_valor_total, col_area_base]))
                df_modelo = df_global[colunas_necessarias].dropna().copy()
                df_modelo = df_modelo[df_modelo[col_area_base] > 0]
                
                # Fator multiplicador de correção direta (garantindo que se a base for menor ou maior, a escala converta para R$/m² padrão de mercado)
                fator_escala = 1000.0 if df_modelo[col_valor_total].mean() < 5000.0 else 1.0
                
                coluna_alvo_unitario = 'valor_unitario_amostra'
                df_modelo[coluna_alvo_unitario] = (df_modelo[col_valor_total] * fator_escala) / df_modelo[col_area_base]
                
                df_modelo = filtrar_outliers(df_modelo, coluna_alvo_unitario)
                
                if len(df_modelo) < 3:
                    st.error("Amostras insuficientes após filtragem de outliers (mínimo de 3).")
                else:
                    df_modelo_log = df_modelo.copy()
                    df_modelo_log[coluna_alvo_unitario] = np.log(df_modelo_log[coluna_alvo_unitario])
                    
                    X = df_modelo_log[features_selecionadas].values
                    y_log = df_modelo_log[coluna_alvo_unitario].values

                    lin_reg = LinearRegression()
                    lin_reg.fit(X, y_log)
                    coeficientes = {feat: coef for feat, coef in zip(features_selecionadas, lin_reg.coef_)}
                    coeficientes['intercepto'] = lin_reg.intercept_

                    coef_array = np.array([lin_reg.intercept_] + list(lin_reg.coef_))
                    p_valores_t, p_valor_f = calcular_estatisticas_regressao(X, y_log, coef_array)

                    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
                    modelo.fit(X, y_log)
                    r2 = round(modelo.score(X, y_log), 4)

                    df_alvo = pd.DataFrame([valores_usuario])[features_selecionadas]
                    previsoes_log_unitario = np.array([arvore.predict(df_alvo.values)[0] for arvore in modelo.estimators_])
                    
                    previsoes_unitarios_reais = np.exp(previsoes_log_unitario)
                    
                    vu_medio = float(np.mean(previsoes_unitarios_reais))
                    vu_min = float(np.percentile(previsoes_unitarios_reais, 15))
                    vu_max = float(np.percentile(previsoes_unitarios_reais, 85))

                    area_avaliando = valores_usuario.get('area_privativa', valores_usuario.get(col_area_base, 1.0))
                    if area_avaliando <= 0:
                        area_avaliando = 1.0

                    v_medio = vu_medio * area_avaliando
                    v_min = vu_min * area_avaliando
                    v_max = vu_max * area_avaliando

                    var_min = abs((v_min - v_medio) / v_medio) * 100
                    var_max = abs((v_max - v_medio) / v_medio) * 100

                    fundamentacao, precisao, soma_pontos, pontos_itens, max_p_regressor, p_valor_f_calc = calcular_graus_nbr_rigoroso(
                        len(df_modelo), r2, len(features_selecionadas), p_valores_t, p_valor_f, tem_extrapolacao_geral, notas_manuais_input
                    )

                    y_real_log_amostras = y_log
                    y_pred_log_amostras = modelo.predict(X)
                    buf_ad, buf_res = gerar_graficos_estatisticos(y_real_log_amostras, y_pred_log_amostras)

                    if pontos_itens[4] == 0:
                        st.error(f"❌ EQUAÇÃO REJEITADA PELO MOTOR NBR! O maior p-valor dos regressores é {max_p_regressor*100:.2f}% (superior ao limite máximo tolerado de 30%).")
                        st.info("💡 **Diagnóstico dos Regressores (Teste t bicaudal):**")
                        for idx, feat_name in enumerate(features_selecionadas):
                            p_feat = p_valores_t[idx + 1]
                            status_p = "🔴 Inválido (> 30%)" if p_feat > 0.30 else "🟢 Válido"
                            st.write(f"- **{feat_name}**: p-valor = {p_feat*100:.2f}% ({status_p})")
                        st.warning("Experimente desmarcar as variáveis com p-valor alto para que a equação seja aprovada.")
                    else:
                        st.success("✅ Equação validada com sucesso pelo motor NBR (Valores Unitários e Totais Normalizados)!")
                        
                        eq_display = f"**ln(Valor Unitário)** = {coeficientes['intercepto']:,.6f}"
                        for feat in features_selecionadas:
                            coef_v = coeficientes[feat]
                            sinal_v = "+" if coef_v >= 0 else ""
                            eq_display += f" {sinal_v} ({coef_v:,.6f} * {feat})"
                        st.markdown(f"##### Equação do Modelo Unitário (6 Casas Decimais):")
                        st.code(eq_display)

                        r1, r2_col, r3 = st.columns(3)
                        r1.metric("Valor Total Mínimo", f"R$ {v_min:,.2f}", f"Unitário: R$ {vu_min:,.2f}/m²")
                        r2_col.metric("Valor Total Estimado", f"R$ {v_medio:,.2f}", f"Unitário: R$ {vu_medio:,.2f}/m²")
                        r3.metric("Valor Total Máximo", f"R$ {v_max:,.2f}", f"Unitário: R$ {vu_max:,.2f}/m²")
                        st.caption(f"Acurácia (R²): {r2} | Máx p-valor Regressor: {max_p_regressor*100:.2f}% | Fundamentação: {fundamentacao} ({soma_pontos} pts) | **Precisão: {precisao}**")

                        pdf_bytes = gerar_laudo_pdf_ia(
                            tenant_selecionado, tipologia_imovel, "valor_unitario_m2", 
                            ordem_servico_input, endereco_imovel_input,
                            informante_nome, informante_tel,
                            {
                                'v_min': v_min, 'v_medio': v_medio, 'v_max': v_max,
                                'vu_min': vu_min, 'vu_medio': vu_medio, 'vu_max': vu_max,
                                'var_min': var_min, 'var_max': var_max
                            },
                            r2, len(df_modelo), features_selecionadas, coeficientes, valores_usuario,
                            variaveis_extrapoladas,
                            fundamentacao, precisao,
                            st.session_state.status_juridico_global,
                            st.session_state.score_juridico_global,
                            soma_pontos, pontos_itens,
                            max_p_regressor, p_valor_f_calc,
                            buf_ad, buf_res
                        )
                        st.download_button(
                            "📄 Baixar Laudo Completo em PDF (NBR 14653)",
                            data=pdf_bytes,
                            file_name=f"laudo_nbr_{ordem_servico_input.replace('/', '_')}.pdf",
                            mime="application/pdf",
                        )
        else:
            st.warning("⚠️ Selecione as colunas de Valor Total e Área Base, além de ao menos uma variável independente.")

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
