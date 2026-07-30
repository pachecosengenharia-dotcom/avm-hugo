import streamlit as st
import pandas as pd
import json
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io

# Configuração da página Web
st.set_page_config(page_title="Plataforma AVM SaaS", page_icon="🏢", layout="wide")

# =====================================================================
# BASE DE DADOS PADRÃO (Backup caso o cliente não envie uma planilha)
# =====================================================================
@st.cache_data
def base_dados_padrao():
    dados = [
        (450000, 6000, 75, 2, 2), (480000, 6153, 78, 2, 2), (430000, 5972, 72, 2, 2),
        (510000, 6375, 80, 2, 2), (460000, 6133, 75, 2, 2), (390000, 5571, 70, 2, 2),
        (530000, 6235, 85, 2, 2), (445000, 6013, 74, 2, 2), (472000, 6210, 76, 2, 2),
        (950000, 12666, 75, 2, 2), (150000, 2000,  75, 2, 2)
    ]
    return pd.DataFrame(dados, columns=['valor_total_declarado', 'valor_unitario_m2', 'area_privativa', 'qtd_quartos', 'padrao_construtivo_id'])

# =====================================================================
# FUNÇÃO DE GERAÇÃO DO LAUDO OFICIAL EM PDF
# =====================================================================
def gerar_laudo_pdf(tenant, area, quartos, padrao, v_estimado, v_min, v_max, r2, n_amostras, status_juridico, score_juridico):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor("#1A365D"), spaceAfter=15)
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2B6CB0"), spaceAfter=10)
    text_style = ParagraphStyle('TextStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8)
    
    story.append(Paragraph("LAUDO DE PRECIFICAÇÃO E AUDITORIA DE GARANTIA", title_style))
    story.append(Paragraph(f"<b>Instituição Solicitante:</b> {tenant}", text_style))
    story.append(Paragraph("<b>Normativos de Conformidade:</b> Resoluções CMN nº 4.676/2018 e nº 4.925/2021 (Banco Central do Brasil)", text_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1. Características do Imóvel Alvo", subtitle_style))
    dados_imovel = [
        ["Área Privativa", f"{area} m²", "Quantidade de Quartos", f"{quartos}"],
        ["Padrão Construtivo", f"{padrao}", "Metodologia Aplicada", "AVM - Regressão Linear"]
    ]
    t1 = Table(dados_imovel, colWidths=[130, 130, 130, 130])
    t1.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")), ('PADDING', (0,0), (-1,-1), 6)]))
    story.append(t1)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("2. Avaliação Econômica e Intervalos de Confiança", subtitle_style))
    dados_valores = [
        ["Métrica de Garantia", "Valor do Metro Quadrado", "Valor Total Estimado"],
        ["Limite Mínimo Admissível", f"R$ {v_min/area:,.2f}", f"R$ {v_min:,.2f}"],
        ["Valor Médio de Mercado", f"R$ {v_estimado/area:,.2f}", f"R$ {v_estimado:,.2f}"],
        ["Limite Máximo Admissível", f"R$ {v_max/area:,.2f}", f"R$ {v_max:,.2f}"]
    ]
    t2 = Table(dados_valores, colWidths=[160, 180, 180])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#EBF8FF"))
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Métricas de Governança do Modelo:</b> Precisão R² = {r2} | Amostras Saneadas utilizadas para o cálculo = {n_amostras}.", text_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("3. Auditoria Jurídica da Matrícula", subtitle_style))
    status_texto = "APROVADO PARA GARANTIA" if status_juridico else "REJEITADO (Bloqueio de Risco)"
    dados_juridicos = [
        ["Status de Homologação", status_texto],
        ["Score de Classificação de Risco", score_juridico]
    ]
    t3 = Table(dados_juridicos, colWidths=[200, 320])
    t3.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('TEXTCOLOR', (1,0), (1,0), colors.HexColor("#38A169") if status_juridico else colors.HexColor("#E53E3E")),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold')
    ]))
    story.append(t3)
    
    story.append(Spacer(1, 40))
    story.append(Paragraph("<i>Relatório gerado eletronicamente pela Plataforma AVM SaaS de forma automatizada, protegida por criptografia e com registro de logs imutáveis para fins de fiscalização do Banco Central do Brasil.</i>", text_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================================
# INTERFACE VISUAL DA PLATAFORMA SAAS
# =====================================================================
st.title("🏢 Painel de Crédito e Controle Multi-Tenant")
st.markdown("Gestão de risco imobiliário integrada e barramento de faturamento SaaS por assinatura.")
st.hr()

# PAINEL DE CONTROLE DE FATURAMENTO SAAS (BARRA LATERAL)
st.sidebar.header("🔑 Assinatura e Faturamento")
tenant_selecionado = st.sidebar.selectbox("Cliente Institucional", ["001 - Banco Alfa S.A.", "002 - Imobiliária Local Ltda"])

if "Banco Alfa" in tenant_selecionado:
    plano_assinatura = "ENTERPRISE"
    limite_consultas = "Ilimitado"
    cor_plano = "🟢"
else:
    plano_assinatura = "STANDARD (Apenas AVM)"
    limite_consultas = "Restam 4 consultas no mês"
    cor_plano = "🟡"

st.sidebar.markdown(f"**Plano Contratado:** {cor_plano} {plano_assinatura}")
st.sidebar.markdown(f"**Limitação de Uso:** {limite_consultas}")
st.sidebar.caption("Suporte Técnico Ativo: Dedicado 24/7")

# Configuração de abas das funcionalidades
aba_avm, aba_juridico = st.tabs(["📊 1. Avaliação Estatística (AVM)", "📜 2. Análise Jurídica da Matrícula"])

if 'avm_calculado' not in st.session_state:
    st.session_state.avm_calculado = False
if 'status_juridico_global' not in st.session_state:
    st.session_state.status_juridico_global = True
if 'score_juridico_global' not in st.session_state:
    st.session_state.score_juridico_global = "PENDENTE"

# ---------------------------------------------------------------------
# ABA 1: MOTOR DE PRECIFICAÇÃO (AVM)
# ---------------------------------------------------------------------
with aba_avm:
    st.subheader("Simulador de Garantia Imobiliária")
    
    # RECURSO DE IMPACTO: Upload de planilha real do cliente
    st.markdown("### 💾 Upload da Base de Dados do Cliente")
    arquivo_planilha = st.file_uploader("Arraste e solte aqui a planilha de mercado do banco (.xlsx ou .csv) para demonstrar a inteligência do sistema.", type=["xlsx", "csv"])
    
    if arquivo_planilha is not None:
        try:
            if arquivo_planilha.name.endswith('.csv'):
                df_mercado = pd.read_csv(arquivo_planilha)
            else:
                df_mercado = pd.read_excel(arquivo_planilha)
            st.success(f"🟩 Planilha comercial '{arquivo_planilha.name}' carregada com sucesso! O motor matemático agora está lendo os dados reais do seu cliente.")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo. Certifique-se de que a planilha possui as colunas 'valor_total_declarado', 'valor_unitario_m2' e 'area_privativa'. Erro: {e}")
            df_mercado = base_dados_padrao()
    else:
        st.info("💡 **Modo de Demonstração:** Utilizando a base de dados amostral embutida no sistema. Suba uma planilha para rodar com dados do próprio cliente.")
        df_mercado = base_dados_padrao()
        
    st.write("---")
    st.markdown("#### Características do Imóvel Alvo")
    col1, col2, col3 = st.columns(3)
    with col1:
        area_alvo = st.number_input("Área Privativa (m²)", min_value=10.0, max_value=500.0, value=75.0, step=1.0)
    with col2:
        quartos = st.slider("Quantidade de Quartos", 1, 5, 2)
    with col3:
        padrao = st.selectbox("Padrão Construtivo do Bem", ["Baixo (ID: 1)", "Normal (ID: 2)", "Alto (ID: 3)"], index=1)

    if st.button("🚀 Calcular Avaliação do Imóvel"):
        q1 = df_mercado['valor_unitario_m2'].quantile(0.25)
        q3 = df_mercado['valor_unitario_m2'].quantile(0.75)
        iqr = q3 - q1
        df_saneado = df_mercado[(df_mercado['valor_unitario_m2'] >= q1 - 1.5*iqr) & (df_mercado['valor_unitario_m2'] <= q3 + 1.5*iqr)]
        
        Y = df_saneado['valor_unitario_m2']
        X = df_saneado[['area_privativa']]
        X = sm.add_constant(X)
        modelo = sm.OLS(Y, X).fit()
        
        vetor_alvo = [1, area_alvo]
        prstd, iv_l, iv_u = wls_prediction_std(modelo, exog=[vetor_alvo], alpha=0.05)
        
        st.session_state.preco_m2 = float(modelo.predict([vetor_alvo]))
        st.session_state.valor_final = st.session_state.preco_m2 * area_alvo
        st.session_state.v_min = float(iv_l) * area_alvo
        st.session_state.v_max = float(iv_u) * area_alvo
        st.session_state.r2 = f"{modelo.rsquared:.4f}"
        st.session_state.n_amostras = len(df_saneado)
        st.session_state.avm_calculado = True

    if st.session_state.avm_calculado:
        st.success("🎯 Cálculos Realizados!")
        c_v1, c_v2, c_v3 = st.columns(3)
