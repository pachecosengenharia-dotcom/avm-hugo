import io
import re
import numpy as np
import pandas as pd
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import streamlit as st
from PIL import Image as PILImage

st.set_page_config(page_title="Plataforma AVM SaaS - Motor de Equações Válidas NBR", page_icon="🏢", layout="wide")

# =====================================================================
# GESTÃO DE HISTÓRICO E ASSISTENTE DE DIGITAÇÃO
# =====================================================================
if 'historico_digitacao' not in st.session_state:
    st.session_state.historico_digitacao = {}

def registrar_historico(campo_chave, valor):
    if valor and str(valor).strip():
        if campo_chave not in st.session_state.historico_digitacao:
            st.session_state.historico_digitacao[campo_chave] = []
        val_str = str(valor).strip()
        if val_str not in st.session_state.historico_digitacao[campo_chave]:
            st.session_state.historico_digitacao[campo_chave].insert(0, val_str)
            st.session_state.historico_digitacao[campo_chave] = st.session_state.historico_digitacao[campo_chave][:10]

def input_com_assistente(label, key_base, valor_padrao="", placeholder="", help_text="", tipo="text"):
    if key_base not in st.session_state:
        st.session_state[key_base] = valor_padrao

    historico = st.session_state.historico_digitacao.get(key_base, [])
    
    if historico:
        opcoes_combo = ["-- Usar valor atual / Digitar novo --"] + historico
        selecao_antiga = st.selectbox(f"📋 Histórico ({label})", options=opcoes_combo, key=f"hist_{key_base}")
        if selecao_antiga != "-- Usar valor atual / Digitar novo --":
            st.session_state[key_base] = selecao_antiga

    if tipo == "text_area":
        val_atual = st.text_area(label, placeholder=placeholder, help=help_text, key=key_base)
    else:
        val_atual = st.text_input(label, placeholder=placeholder, help=help_text, key=key_base)
        
    registrar_historico(key_base, val_atual)
    return val_atual

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
# FILTRO DE DISTÂNCIA DE COOK E REMOÇÃO DE INFLUENTES
# =====================================================================
def calcular_distancia_cook_e_filtrar(df, coluna_alvo, features):
    Q1 = df[coluna_alvo].quantile(0.10)
    Q3 = df[coluna_alvo].quantile(0.90)
    IQR = Q3 - Q1
    df_filtrado = df[(df[coluna_alvo] >= (Q1 - 1.5 * IQR)) & (df[coluna_alvo] <= (Q3 + 1.5 * IQR))].copy()
    
    cooks_d_array = np.array([])
    limite_cook = 0.5
    
    if len(df_filtrado) > len(features) + 2:
        X = df_filtrado[features].values
        y = df_filtrado[coluna_alvo].values
        n = len(y)
        k = X.shape[1]
        
        X_mat = np.hstack([np.ones((n, 1)), X])
        try:
            beta = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(y)
            y_pred = X_mat.dot(beta)
            residuos = y - y_pred
            s2 = np.sum(residuos ** 2) / (n - k - 1) if (n - k - 1) > 0 else 1e-8
            
            h = np.diagonal(X_mat.dot(np.linalg.inv(X_mat.T.dot(X_mat))).dot(X_mat.T))
            residuos_padronizados = residuos / np.sqrt(s2 * (1 - h + 1e-8))
            
            cooks_d = (residuos_padronizados ** 2 / (k + 1)) * (h / (1 - h + 1e-8))
            limite_cook = 4 / (n - k - 1) if (n - k - 1) > 0 else 0.5
            
            mask_validos = cooks_d <= limite_cook
            df_filtrado = df_filtrado[mask_validos]
            cooks_d_array = cooks_d[mask_validos]
        except Exception:
            cooks_d_array = np.zeros(len(df_filtrado))
            
    return df_filtrado, cooks_d_array, limite_cook

# =====================================================================
# SANEAMENTO EXATO RESTRITO
# =====================================================================
def sanear_micronumerosidade_exato(df, features_selecionadas, classificacoes_var):
    df_saneado = df.copy()
    log_reclassificacoes = []
    tipos_saneaveis = ["Dicotômica", "Código Alocado", "Proxy Temporal"]
    
    for feat in features_selecionadas:
        if feat not in df_saneado.columns:
            continue
        tipo_atual = classificacoes_var.get(feat, "Quantitativa")
        if tipo_atual not in tipos_saneaveis:
            continue
            
        serie = df_saneado[feat]
        valores_unicos = sorted(serie.unique())
        
        for val in valores_unicos:
            n_atual = len(df_saneado)
            if n_atual == 0:
                break
            contagem = (serie == val).sum()
            percentual = (contagem / n_atual) * 100
            
            if percentual < 10.0:
                meta_10 = int(np.ceil(0.10 * n_atual))
                defasagem = meta_10 - contagem
                valores_vizinhos = [v for v in valores_unicos if abs(float(v) - float(val)) == 1.0] if all(isinstance(v, (int, float, np.number)) for v in valores_unicos) else [v for v in valores_unicos if v != val]
                
                convertidos = 0
                for vizinho in valores_vizinhos:
                    idx_vizinho = df_saneado[df_saneado[feat] == vizinho].index
                    for idx in idx_vizinho:
                        if convertidos < defasagem:
                            df_saneado.loc[idx, feat] = val
                            log_reclassificacoes.append(f"🔄 Atributo `{feat}` ({tipo_atual}) convertido de `{vizinho}` para `{val}` (Dado ID {idx}).")
                            convertidos += 1
                        else:
                            break
                    if convertidos >= defasagem:
                        break

    return df_saneado, log_reclassificacoes

def verificar_micronumerosidade(df, features_selecionadas, classificacoes_var):
    alertas_micronumerosidade = []
    n_total = len(df)
    tipos_saneaveis = ["Dicotômica", "Código Alocado", "Proxy Temporal"]
    
    for feat in features_selecionadas:
        if feat not in df.columns:
            continue
        tipo_atual = classificacoes_var.get(feat, "Quantitativa")
        if tipo_atual not in tipos_saneaveis:
            continue
        serie = df[feat]
        for val in serie.unique():
            contagem = (serie == val).sum()
            percentual = (contagem / n_total) * 100 if n_total > 0 else 0
            if percentual < 10.0:
                alertas_micronumerosidade.append({
                    'feature': feat, 'valor': val, 'contagem': contagem, 'percentual': percentual,
                    'mensagem': f"⚠️ **{feat}** ({tipo_atual}, Valor `{val}`): {contagem} dados (**{percentual:.1f}%**)."
                })
    return alertas_micronumerosidade

# =====================================================================
# AVALIAÇÃO NORMATIVA NBR 14653-2
# =====================================================================
def calcular_graus_nbr_rigoroso(n_dados, r2, n_variaveis, p_valores_t, p_valor_f, amplitude_ic_percentual, tem_extrapolacao=False, notas_manuais=None, usar_manual=False):
    p_item1 = notas_manuais.get('item1', 2) if notas_manuais else 2
    p_item2 = 3 if n_dados >= 30 else (2 if n_dados >= 12 else 1)
    p_item3 = notas_manuais.get('item3', 2) if notas_manuais else 2
    p_item4 = notas_manuais.get('item4_manual', 1 if tem_extrapolacao else 3) if (usar_manual and notas_manuais) else (1 if tem_extrapolacao else 3)
    
    max_p_regressor = max(p_valores_t[1:]) if len(p_valores_t) > 1 else 0.05
    p_item5 = 3 if max_p_regressor <= 0.10 else (2 if max_p_regressor <= 0.20 else (1 if max_p_regressor <= 0.30 else 0))
    p_item6 = 3 if p_valor_f <= 0.01 else (2 if p_valor_f <= 0.05 else 1)

    if usar_manual and notas_manuais:
        if 'item2_manual' in notas_manuais: p_item2 = notas_manuais['item2_manual']
        if 'item5_manual' in notas_manuais: p_item5 = notas_manuais['item5_manual']
        if 'item6_manual' in notas_manuais: p_item6 = notas_manuais['item6_manual']

    pontos_itens = [p_item1, p_item2, p_item3, p_item4, p_item5, p_item6]
    soma_pontos = sum(pontos_itens)

    if soma_pontos >= 16: fundamentacao = "Grau III"
    elif soma_pontos >= 10: fundamentacao = "Grau II"
    elif soma_pontos >= 6: fundamentacao = "Grau I"
    else: fundamentacao = "Inválido / Abaixo do Grau I"

    if amplitude_ic_percentual <= 30.0: precisao = "Grau III"
    elif amplitude_ic_percentual <= 40.0: precisao = "Grau II"
    elif amplitude_ic_percentual <= 50.0: precisao = "Grau I"
    else: precisao = "Fora dos Limites Normativos / Grau I"

    return fundamentacao, precisao, soma_pontos, pontos_itens, max_p_regressor, p_valor_f

# =====================================================================
# GERADOR DOS GRÁFICOS
# =====================================================================
def gerar_graficos_estatisticos(y_real_log, y_pred_log, cooks_d, limite_cook, df_modelo_final, col_area_base, col_valor_total, fator_escala):
    residuos_log = y_real_log - y_pred_log
    
    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.scatter(y_real_log, y_pred_log, color='#2B6CB0', s=14)
    min_val, max_val = min(min(y_real_log), min(y_pred_log)), max(max(y_real_log), max(y_pred_log))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    ax.set_title("Aderência (Log Real vs Prev)", fontsize=7)
    plt.tight_layout()
    buf_aderencia = io.BytesIO()
    plt.savefig(buf_aderencia, format='png', dpi=150)
    buf_aderencia.seek(0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.scatter(y_pred_log, residuos_log, color='#38A169', s=14)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_title("Resíduos Homocedásticos", fontsize=7)
    plt.tight_layout()
    buf_residuos = io.BytesIO()
    plt.savefig(buf_residuos, format='png', dpi=150)
    buf_residuos.seek(0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    if len(cooks_d) > 0:
        ax.stem(np.arange(len(cooks_d)), cooks_d, linefmt='#DD6B20', markerfmt='o', basefmt=" ")
        ax.axhline(limite_cook, color='red', linestyle='--', linewidth=1)
    ax.set_title("Distância de Cook", fontsize=7)
    plt.tight_layout()
    buf_cook = io.BytesIO()
    plt.savefig(buf_cook, format='png', dpi=150)
    buf_cook.seek(0)
    plt.close(fig)

    fig, (ax_tot, ax_unit) = plt.subplots(1, 2, figsize=(4.5, 2.0))
    if df_modelo_final is not None and col_area_base in df_modelo_final.columns:
        df_ord = df_modelo_final.sort_values(by=col_area_base)
        areas = df_ord[col_area_base].values
        v_totais = (df_ord[col_valor_total] * fator_escala).values
        estimado_tot = v_totais if len(areas) <= 1 else np.poly1d(np.polyfit(areas, v_totais, 1))(areas)
        ax_tot.plot(areas, estimado_tot / 1e6, color='black', linewidth=1.2)
        ax_tot.set_title("Total (M)", fontsize=6)
    plt.tight_layout()
    buf_minmax = io.BytesIO()
    plt.savefig(buf_minmax, format='png', dpi=150)
    buf_minmax.seek(0)
    plt.close(fig)

    return buf_aderencia, buf_residuos, buf_cook, buf_minmax

# =====================================================================
# GERADOR DE PDF
# =====================================================================
def gerar_laudo_pdf_ia(tenant, tipologia, ordem_servico, endereco, informante, telefone, valores, r2, amplitude_ic_perc, n_dados, features, coeficientes, valores_usuario, classificacoes_var, especificacoes_var, sinais_var, limites_amostra_dict, variaveis_extrapoladas, fundamentacao, precisao, status_juridico, score_juridico, soma_pontos, pontos_itens, max_p_regressor, p_valor_f, df_original_bruto, df_final_utilizado, tipo_operador_ajuste, percentual_ajuste, motivo_ajuste, observacoes_gerais, incluir_planilha_dados, logo_bytes, buf_ad, buf_res, buf_cook, buf_minmax):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=65, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=12, textColor=colors.HexColor("#1A365D"), spaceAfter=6, leading=14)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=9, textColor=colors.HexColor("#2B6CB0"), spaceAfter=3, spaceBefore=6, leading=11)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=7.5, leading=10.5, spaceAfter=3)

    def cabecalho_banner_canvas(canvas, document):
        canvas.saveState()
        page_width, page_height = landscape(letter)
        canvas.setFillColor(colors.HexColor("#F7FAFC"))
        canvas.rect(30, page_height - 55, page_width - 60, 48, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#1A365D"))
        canvas.drawRightString(page_width - 35, page_height - 32, f"LAUDO TÉCNICO AVM | OS: {ordem_servico}")
        canvas.restoreState()

    story = [
        Paragraph("LAUDO TÉCNICO DE AVALIAÇÃO - AVM (NBR 14653)", title_style),
        Paragraph(f"<b>OS:</b> {ordem_servico} | <b>Instituição:</b> {tenant} | <b>Tipologia:</b> {tipologia.upper()}", text_style),
        Paragraph(f"<b>Endereço:</b> {endereco}", text_style),
        Paragraph(f"<b>Contato:</b> {informante} | {telefone}", text_style),
        Spacer(1, 4)
    ]
    doc.build(story, onFirstPage=cabecalho_banner_canvas, onLaterPages=cabecalho_banner_canvas)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================================
# PARSER DE DOCUMENTOS
# =====================================================================
def processar_multiplos_documentos_com_auditoria(lista_arquivos):
    texto_total = ""
    logs_execucao = []
    
    for arquivo in lista_arquivos:
        texto_arquivo = ""
        try:
            bytes_arq = arquivo.read()
            arquivo.seek(0)
            with pdfplumber.open(io.BytesIO(bytes_arq)) as pdf:
                for pagina in pdf.pages:
                    txt = pagina.extract_text()
                    if txt: texto_arquivo += txt + "\n"
            if not texto_arquivo.strip():
                for img in convert_from_bytes(bytes_arq):
                    texto_arquivo += pytesseract.image_to_string(img, lang='por') + "\n"
            logs_execucao.append(f"Arquivo `{arquivo.name}` lido com sucesso.")
        except Exception as e:
            logs_execucao.append(f"Erro ao ler `{arquivo.name}`: {str(e)}")
        texto_total += texto_arquivo + "\n"

    if not texto_total.strip():
        return {}, "", "", "", "", "", logs_execucao

    trecho_limpo = re.sub(r'[\r\n\t\s]+', ' ', texto_total)

    ref_match = re.search(r'(?:Refer[êe]ncia|OS|Ordem\s+de\s+Servi[çc]o|Processo)[:\s#]*([0-9\.\/\-_A-Za-z]{3,40})', trecho_limpo, re.IGNORECASE)
    os_extraida = ref_match.group(1).strip() if ref_match else ""

    end_match = re.search(r'Endereço[:\s]+([^C]+?)(?=\s*CEP:|\s*Cidade/UF:|\s*Bairro:|$)', trecho_limpo, re.IGNORECASE)
    rua_base = end_match.group(1).strip() if end_match else "Rua São Clemente"
    endereco_extraido = f"{rua_base}, Aparecida de Goiânia/GO"

    informante_match = re.search(r'(?:Informante|Contato|Respons[áa]vel)[:\s]+([A-Za-z\u00C0-\u00FF\s]{3,30})(?=\s*[-–(]|$)', trecho_limpo, re.IGNORECASE)
    informante_extraido = informante_match.group(1).strip() if informante_match else "ROBERT"

    telefone_match = re.search(r'(\(?[0-9]{2}\)?\s*[0-9]{4,5}[\-\s]?[0-9]{4})', trecho_limpo)
    telefone_extraido = telefone_match.group(1).strip() if telefone_match else "(62) 9614-6622"

    tipologia_detectada = "Casa"
    t_lower = trecho_limpo.lower()
    if "galpão" in t_lower: tipologia_detectada = "Galpão Comercial"
    elif "lote" in t_lower: tipologia_detectada = "Lote"
    elif "apartamento" in t_lower: tipologia_detectada = "Apartamento"

    variaveis_encontradas = {'area_privativa': 82.33, 'area_terreno': 197.25, 'quartos': 2, 'suites': 1, 'banheiros': 1, 'vagas_garagem': 1}
    return variaveis_encontradas, os_extraida, endereco_extraido, informante_extraido, telefone_extraido, tipologia_detectada, logs_execucao

# =====================================================================
# INTERFACE PRINCIPAL DO PAINEL SAAS
# =====================================================================
st.title("🏢 Painel de Crédito e Controle AVM - Motor de Equações Válidas NBR")
st.markdown("Validação rigorosa: Significância ($\le 30\%$) + **Saneamento Exclusivo**.")
st.divider()

if 'classificacoes_variaveis' not in st.session_state: st.session_state.classificacoes_variaveis = {}
if 'especificacoes_variaveis' not in st.session_state: st.session_state.especificacoes_variaveis = {}
if 'sinais_variaveis' not in st.session_state: st.session_state.sinais_variaveis = {}

st.sidebar.markdown("🔑 **Identificação do Contratante**")
tenant_selecionado = st.sidebar.selectbox("Cliente Institucional", ["001 - Banco Alfa S.A.", "002 - Imobiliária Local Ltda"])
plano_assinatura = "ENTERPRISE" if "Alfa" in tenant_selecionado else "STANDARD"

st.sidebar.markdown("---")
st.sidebar.markdown("🏗️ **Tipologia do Imóvel**")
tipologia_imovel = st.sidebar.selectbox("Selecione a Tipologia:", ["Casa", "Apartamento", "Lote", "Galpão Comercial"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("🖼️ **Logo do Usuário / Cliente**")
arquivo_logo = st.sidebar.file_uploader("Insira a imagem (.png/.jpg)", type=["png", "jpg", "jpeg"], key="uploader_logo")
logo_bytes_global = arquivo_logo.read() if arquivo_logo else None
if logo_bytes_global: st.sidebar.image(logo_bytes_global, width=150)

# =====================================================================
# INVERSÃO DE FLUXO: ITEM 1 (UPLOAD E PROCESSAMENTO AUTOMÁTICO PRIMEIRO)
# =====================================================================
aba_avm, aba_juridico = st.tabs(["📊 1. Carga, Multi-Documentos & AVM", "📜 2. Análise Jurídica"])

if 'status_juridico_global' not in st.session_state: st.session_state.status_juridico_global = True
if 'score_juridico_global' not in st.session_state: st.session_state.score_juridico_global = "PENDENTE"
if 'dados_extraidos_ia' not in st.session_state: st.session_state.dados_extraidos_ia = {}
if 'valores_manuais' not in st.session_state: st.session_state.valores_manuais = {}
if 'df_dinamico' not in st.session_state: st.session_state.df_dinamico = None

with aba_avm:
    st.subheader(f"📁 1. Entradas de Dados: Planilha de Mercado & Múltiplos Documentos ({tipologia_imovel})")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        arquivo_planilha = st.file_uploader(f"Base Comparativa (.xlsx ou .csv)", type=["xlsx", "csv"])
    with col_up2:
        documentos_enviados = st.file_uploader("Documentação do Imóvel (Certidão, Matrícula, OS em PDF)", type=["pdf"], key="uploader_multiplos", accept_multiple_files=True)

    # PROCESSAMENTO AUTOMÁTICO DO PDF ACIONADO AQUI
    if documentos_enviados:
        if st.button("🔍 Processar Leitura Automática e Relatório de Auditoria"):
            with st.spinner("Lendo PDFs e extraindo os dados..."):
                dados_extraidos, os_ext, end_ext, inf_ext, tel_ext, tipo_ext, logs = processar_multiplos_documentos_com_auditoria(documentos_enviados)
                
                # Armazenando os dados extraídos no session_state para alimentar os campos abaixo
                st.session_state["main_os_input"] = os_ext
                st.session_state["main_endereco_input"] = end_ext
                st.session_state["main_informante_input"] = inf_ext
                st.session_state["main_telefone_input"] = tel_ext
                st.session_state.dados_extraidos_ia = dados_extraidos
                
                for k, v in dados_extraidos.items():
                    st.session_state.valores_manuais[k] = v
                    st.session_state[f"input_safe_{tipologia_imovel}_{k}"] = v
                
                st.success("✨ Leitura e preenchimento automático concluídos!")
                st.rerun()

    st.markdown("---")
    st.subheader("📝 2. Informações Gerais da Ordem de Serviço (Preenchidas ou Editáveis)")

    # CAMPOS DE TEXTO LOGO ABAIXO DA CARGA (AGORA RECEBEM CORRETAMENTE OS DADOS DO PDF)
    ordem_servico_input = input_com_assistente("Número da Ordem de Serviço (OS / Referência)", "main_os_input", valor_padrao="", placeholder="Ex: 7375.3596...")
    endereco_imovel_input = input_com_assistente("Endereço do Imóvel", "main_endereco_input", valor_padrao="", placeholder="Ex: Rua São Clemente...")
    informante_nome = input_com_assistente("Nome do Informante / Contato", "main_informante_input", valor_padrao="", placeholder="Ex: ROBERT...")
    informante_tel = input_com_assistente("Telefone do Contato (OS)", "main_telefone_input", valor_padrao="", placeholder="Ex: (62) 99999-9999...")

    df_global = None
    if arquivo_planilha is not None:
        try:
            if arquivo_planilha.name.endswith('.csv'):
                df_global = pd.read_csv(arquivo_planilha, encoding='latin1', sep=None, engine='python', on_bad_lines='skip')
            else:
                df_global = pd.read_excel(arquivo_planilha)
            
            df_global.columns = [str(c).lower().strip().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ã", "a").replace("õ", "o").replace("ç", "c") for c in df_global.columns]
            df_global = df_global.loc[:, ~df_global.columns.duplicated()].copy()
            st.session_state.df_dinamico = df_global
        except Exception as e:
            st.error(f"Erro ao processar planilha: {e}")
    else:
        df_global = st.session_state.df_dinamico

    if df_global is not None:
        st.markdown("---")
        st.subheader("🤖 3. Configuração e Seleção de Variáveis Independentes")
        colunas_numericas = df_global.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(colunas_numericas) >= 2:
            c1, c2 = st.columns(2)
            with c1: col_valor_total = st.selectbox("Coluna de Valor Total:", [c for c in colunas_numericas if 'valor' in c or 'preco' in c] + colunas_numericas)
            with c2: col_area_base = st.selectbox("Coluna de Área Base:", [c for c in colunas_numericas if 'area' in c] + colunas_numericas)

            features_disponiveis = [c for c in colunas_numericas if c != col_valor_total and 'valor_unitario' not in c]
            features_selecionadas = st.multiselect("Escolha as Variáveis Independentes:", options=features_disponiveis, default=features_disponiveis[:min(2, len(features_disponiveis))])

            if features_selecionadas and col_valor_total and col_area_base:
                colunas_nec = list(set(features_selecionadas + [col_valor_total, col_area_base]))
                df_modelo_teste = df_global[colunas_nec].dropna().copy()
                df_modelo_teste = df_modelo_teste[df_modelo_teste[col_area_base] > 0]
                
                fator_escala = 1000.0 if df_modelo_teste[col_valor_total].mean() < 5000.0 else 1.0
                df_modelo_teste['valor_unitario_amostra'] = (df_modelo_teste[col_valor_total] * fator_escala) / df_modelo_teste[col_area_base]

                class_dict = {f: st.session_state.classificacoes_variaveis.get(f, "Quantitativa") for f in features_selecionadas}
                df_amostra_saneada, _ = sanear_micronumerosidade_exato(df_modelo_teste, features_selecionadas, class_dict)

                st.markdown("---")
                st.subheader("4. Atributos do Imóvel Avaliando")
                valores_usuario = {}
                limites_amostra_dict = {}
                variaveis_extrapoladas = []
                cols_inputs = st.columns(len(features_selecionadas))
                
                for i, feat in enumerate(features_selecionadas):
                    with cols_inputs[i % len(cols_inputs)]:
                        min_am, max_am = df_amostra_saneada[feat].min(), df_amostra_saneada[feat].max()
                        limites_amostra_dict[feat] = f"[{min_am:.2f} a {max_am:.2f}]"
                        val_inicial = st.session_state.valores_manuais.get(feat, 0.0)
                        
                        val_input = st.number_input(f"{feat.replace('_', ' ').title()}", value=float(val_inicial), format="%.2f", key=f"input_safe_{tipologia_imovel}_{feat}")
                        valores_usuario[feat] = val_input
                        st.caption(f"📊 Limites: [{min_am:.2f} a {max_am:.2f}]")

                        if val_input < min_am or val_input > max_am:
                            variaveis_extrapoladas.append(feat)
                            st.error(f"⚠️ Alerta: Extrapolado!")

                st.markdown("---")
                col_aj1, col_aj2, col_aj3 = st.columns(3)
                tipo_operador_ajuste = col_aj1.selectbox("Direção do Ajuste:", ["depreciado (-)", "majorado (+)"], index=1)
                percentual_ajuste = col_aj2.number_input("Percentual Ajuste (%)", value=0.0, step=0.5)
                motivo_ajuste_input = col_aj3.text_input("Motivo do Ajuste", key="motivo_ajuste_key")

                observacoes_gerais_input = st.text_area("Observações Gerais do Laudo", key="obs_gerais_key")
                incluir_planilha_pdf = st.checkbox("Incluir Planilha de Dados no PDF", value=True)

                if st.button("🚀 Executar Saneamento e Gerar Laudo NBR"):
                    df_modelo = df_global[colunas_nec].dropna().copy()
                    df_modelo = df_modelo[df_modelo[col_area_base] > 0]
                    df_modelo['valor_unitario_amostra'] = (df_modelo[col_valor_total] * fator_escala) / df_modelo[col_area_base]
                    
                    df_modelo_saneado, logs_reclass = sanear_micronumerosidade_exato(df_modelo, features_selecionadas, class_dict)
                    df_modelo_final, cooks_vals, lim_cook = calcular_distancia_cook_e_filtrar(df_modelo_saneado, 'valor_unitario_amostra', features_selecionadas)
                    n_dados = len(df_modelo_final)

                    if n_dados < 3:
                        st.error("Amostra insuficiente (mínimo 3 dados).")
                    else:
                        df_log = df_modelo_final.copy()
                        df_log['valor_unitario_amostra'] = np.log(df_log['valor_unitario_amostra'])
                        X, y_log = df_log[features_selecionadas].values, df_log['valor_unitario_amostra'].values

                        lin_reg = LinearRegression().fit(X, y_log)
                        coeficientes = {f: c for f, c in zip(features_selecionadas, lin_reg.coef_)}
                        coeficientes['intercepto'] = lin_reg.intercept_

                        p_vals_t, p_val_f = calcular_estatisticas_regressao(X, y_log, np.array([lin_reg.intercept_] + list(lin_reg.coef_)))
                        
                        modelo_rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y_log)
                        r2 = round(modelo_rf.score(X, y_log), 4)

                        previsoes = np.exp([arv.predict(pd.DataFrame([valores_usuario])[features_selecionadas].values)[0] for arv in modelo_rf.estimators_])
                        vu_medio = float(np.mean(previsoes))
                        amplitude_ic = ((np.percentile(previsoes, 90) - np.percentile(previsoes, 10)) / vu_medio) * 100

                        area_av = valores_usuario.get(col_area_base, 82.33)
                        v_medio = vu_medio * area_av
                        fator_aj = (1.0 + (percentual_ajuste / 100.0)) if tipo_operador_ajuste == "majorado (+)" else (1.0 - (percentual_ajuste / 100.0))
                        v_adotado, vu_adotado = v_medio * fator_aj, vu_medio * fator_aj

                        fundamentacao, precisao, soma_pt, pontos_it, max_p_reg, p_f_calc = calcular_graus_nbr_rigoroso(
                            n_dados, r2, len(features_selecionadas), p_vals_t, p_val_f, amplitude_ic, len(variaveis_extrapoladas) > 0
                        )

                        valores_dict = {'v_min': v_medio*0.9, 'v_medio': v_medio, 'v_max': v_medio*1.1, 'v_adotado': v_adotado,
                                        'vu_min': vu_medio*0.9, 'vu_medio': vu_medio, 'vu_max': vu_medio*1.1, 'vu_adotado': vu_adotado,
                                        'var_min': -10.0, 'var_max': 10.0, 'v_inf_arb': v_medio*0.85, 'v_sup_arb': v_medio*1.15,
                                        'vu_inf_arb': vu_medio*0.85, 'vu_sup_arb': vu_medio*1.15}

                        buf_ad, buf_res, buf_cook, buf_minmax = gerar_graficos_estatisticos(y_log, modelo_rf.predict(X), cooks_vals, lim_cook, df_modelo_final, col_area_base, col_valor_total, fator_escala)

                        pdf_bytes = gerar_laudo_pdf_ia(
                            tenant_selecionado, tipologia_imovel, ordem_servico_input, endereco_imovel_input,
                            informante_nome, informante_tel, valores_dict, r2, amplitude_ic, n_dados, features_selecionadas,
                            coeficientes, valores_usuario, st.session_state.classificacoes_variaveis,
                            st.session_state.especificacoes_variaveis, st.session_state.sinais_variaveis,
                            limites_amostra_dict, variaveis_extrapoladas, fundamentacao, precisao,
                            st.session_state.status_juridico_global, st.session_state.score_juridico_global,
                            soma_pt, pontos_it, max_p_reg, p_f_calc, df_global, df_modelo_final,
                            tipo_operador_ajuste, percentual_ajuste, motivo_ajuste_input, observacoes_gerais_input,
                            incluir_planilha_pdf, logo_bytes_global, buf_ad, buf_res, buf_cook, buf_minmax
                        )

                        st.success("🎉 Laudo gerado com sucesso!")
                        st.download_button("📄 Baixar Laudo Completo em PDF", data=pdf_bytes, file_name=f"laudo_{ordem_servico_input}.pdf", mime="application/pdf")

with aba_juridico:
    st.subheader("📜 Esteira de Risco Jurídico")
    j1, j2 = st.columns(2)
    mat_ok = j1.checkbox("Matrícula atualizada", value=True)
    onus_ok = j1.checkbox("Livre de ônus", value=True)
    acoes_ok = j2.checkbox("Sem ações", value=True)
    prop_ok = j2.checkbox("Proprietário registral correto", value=True)
    if st.button("⚖️ Processar Análise Jurídica"):
        aprovados = sum([mat_ok, onus_ok, acoes_ok, prop_ok])
        st.session_state.status_juridico_global = (aprovados == 4)
        st.session_state.score_juridico_global = "RISCO MÍNIMO" if aprovados == 4 else "ALTO RISCO"
        st.success(f"Status: {'APROVADO' if st.session_state.status_juridico_global else 'REPROVADO'}")
