import io
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

st.set_page_config(page_title="Plataforma AVM SaaS - Multi-Tipologia", page_icon="🏢", layout="wide")

# =====================================================================
# SIMULADOR DE GERADOR DE BANCO DE DADOS INTEGRADO
# =====================================================================
@st.cache_data
def carregar_base_multitipologia_padrao():
    dados = [
        (450000, 6000, 75, 1200, 200, 2, 0, 3.0, "CASA"),
        (480000, 6153, 78, 1250, 220, 2, 0, 3.0, "CASA"),
        (510000, 6375, 80, 1300, 250, 2, 0, 3.2, "CASA"),
        (750000, 8823, 85, 3200, 360, 3, 0, 3.5, "CASA"),
        (820000, 8913, 92, 3300, 400, 3, 0, 3.5, "CASA"),
        (350000, 5833, 60, 1500, 0, 1, 3, 2.7, "APARTAMENTO"),
        (380000, 6129, 62, 1600, 0, 1, 5, 2.7, "APARTAMENTO"),
        (420000, 6461, 65, 1800, 0, 2, 8, 2.8, "APARTAMENTO"),
        (650000, 8666, 75, 3400, 0, 2, 12, 2.9, "APARTAMENTO"),
        (720000, 9000, 80, 3600, 0, 2, 15, 3.0, "APARTAMENTO"),
        (150000, 428, 350, 800, 350, 0, 0, 0, "LOTE"),
        (165000, 458, 360, 850, 360, 0, 0, 0, "LOTE"),
        (210000, 525, 400, 1100, 400, 0, 0, 0, "LOTE"),
        (450000, 900, 500, 2800, 500, 0, 0, 0, "LOTE"),
        (1200000, 2400, 500, 900, 800, 0, 0, 6.0, "GALPAO"),
        (1500000, 2500, 600, 950, 900, 0, 0, 7.0, "GALPAO"),
        (2100000, 2625, 800, 1100, 1200, 0, 0, 8.0, "GALPAO"),
    ]
    return pd.DataFrame(dados, columns=[
        'valor_total_declarado', 'valor_unitario_m2', 'area_privativa',
        'indice_fiscal', 'area_terreno', 'vagas_garagem', 'andar', 'pe_direito', 'tipologia'
    ])

# =====================================================================
# GERADOR DE PDF CUSTOMIZADO PARA MULTI-TIPOLOGIA
# =====================================================================
def gerar_laudo_pdf_ia(tenant, tipologia, area, valores, r2, n_amostras, status_juridico, score_juridico):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#1A365D"), spaceAfter=15)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#2B6CB0"), spaceAfter=8)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=6)

    story.append(Paragraph(f"LAUDO CORE AVM - INTELIGÊNCIA ARTIFICIAL ({tipologia})", title_style))
    story.append(Paragraph(f"<b>Instituição Solicitante:</b> {tenant}", text_style))
    story.append(Paragraph("<b>Metodologia Core:</b> Algoritmo de Árvores de Decisão (Random Forest Regressor) | NBR 14653", text_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Escopo de Avaliação Imobiliária", subtitle_style))
    t1 = Table([["Tipologia do Bem", tipologia, "Dimensão Principal", f"{area} m²"]], colWidths=[130, 130, 130, 130])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)

    story.append(Paragraph("2. Resultados do Motor de Machine Learning", subtitle_style))
    t2 = Table([
        ["Métrica de Cobertura do Risco", "Valor Comercial Admissível"],
        ["Margem Mínima de Segurança (Garantia)", f"R$ {valores['v_min']:,.2f}"],
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
    story.append(Paragraph(f"<b>Métricas de Redes de Decisão:</b> Precisão de Treino R² = {r2} | Amostras Saneadas = {n_amostras}.", text_style))

    story.append(Paragraph("3. Status da Esteira de Risco Jurídico", subtitle_style))
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
# MOTOR DE IA (RANDOM FOREST)
# =====================================================================
def executar_motor_ia(df_global, tipologia, area, indice_fiscal, atributos):
    df_global.columns = [
        c.lower().strip().replace(" ", "_")
        .replace("á", "a").replace("í", "i").replace("é", "e")
        .replace("ó", "o").replace("ã", "a").replace("ç", "c")
        for c in df_global.columns
    ]
    
    df_tipo = df_global[df_global['tipologia'] == tipologia].copy()
    features = ['area_privativa', 'indice_fiscal', 'area_terreno', 'vagas_garagem', 'andar', 'pe_direito']
    
    faltantes = [f for f in (features + ['valor_total_declarado']) if f not in df_tipo.columns]
    if faltantes:
        return None, f"Colunas ausentes no arquivo: {', '.join(faltantes)}. Verifique seu cabeçalho."

    if len(df_tipo) < 3:
        return None, "Amostras insuficientes (mínimo 3) para esta tipologia."

    df_tipo = df_tipo.dropna(subset=features + ['valor_total_declarado'])
    
    X = df_tipo[features]
    y = df_tipo['valor_total_declarado']

    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo.fit(X, y)
    r2 = round(modelo.score(X, y), 4)

    alvo = pd.DataFrame([{
        'area_privativa': area,
        'indice_fiscal': indice_fiscal,
        'area_terreno': atributos.get('area_terreno', 0),
        'vagas_garagem': atributos.get('vagas_garagem', 0),
        'andar': atributos.get('andar', 0),
        'pe_direito': atributos.get('pe_direito', 0),
    }])

    previsoes = np.array([arvore.predict(alvo.values)[0] for arvore in modelo.estimators_])
    v_medio = float(np.mean(previsoes))
    v_min = float(np.percentile(previsoes, 15))
    v_max = float(np.percentile(previsoes, 85))
    v_min = min(v_min, v_medio)
    v_max = max(v_max, v_medio)

    return {
        'valores': {'v_min': v_min, 'v_medio': v_medio, 'v_max': v_max},
        'r2': r2,
        'n_amostras': len(df_tipo),
    }, None

# =====================================================================
# INTERFACE PRINCIPAL DO PAINEL SAAS
# =====================================================================
st.title("🏢 Painel de Crédito e Controle Multi-Tenant - Inteligência Artificial")
st.markdown("Gestão automatizada de risco imobiliário horizontal, vertical e industrial.")
st.divider()

st.sidebar.header("🔑 Assinatura e Faturamento")
tenant_selecionado = st.sidebar.selectbox("Cliente Institucional", ["001 - Banco Alfa S.A.", "002 - Imobiliária Local Ltda"])
plano_assinatura = "ENTERPRISE" if "Alfa" in tenant_selecionado else "STANDARD"
st.sidebar.markdown(f"**Plano Contratado:** {'🟢 ENTERPRISE' if plano_assinatura == 'ENTERPRISE' else '🟡 STANDARD'}")

aba_avm, aba_juridico = st.tabs(["📊 1. Avaliação Estatística por IA (AVM)", "📜 2. Análise Jurídica da Matrícula"])

if 'status_juridico_global' not in st.session_state:
    st.session_state.status_juridico_global = True
if 'score_juridico_global' not in st.session_state:
    st.session_state.score_juridico_global = "PENDENTE"

with aba_avm:
    st.subheader("Configuração da Base e Modelagem")
    arquivo_planilha = st.file_uploader("Arraste aqui a planilha consolidada de imóveis do banco (.xlsx ou .csv)", type=["xlsx", "csv"])

    if arquivo_planilha is not None:
        try:
            if arquivo_planilha.name.endswith('.csv'):
                # Adicionado encoding, detecção de separador e tratamento de linhas incorretas
                df_global = pd.read_csv(
                    arquivo_planilha, 
                    encoding='latin1', 
                    sep=None, 
                    engine='python', 
                    on_bad_lines='skip'
                )
            else:
                df_global = pd.read_excel(arquivo_planilha)
            
            df_global.columns = [
                c.lower().strip().replace(" ", "_")
                .replace("á", "a").replace("í", "i").replace("é", "e")
                .replace("ó", "o").replace("ã", "a").replace("ç", "c")
                for c in df_global.columns
            ]

            mapeamento = {
                'valor_total': 'valor_total_declarado',
                'valor': 'valor_total_declarado',
                'area': 'area_privativa',
                'area_privativa': 'area_privativa',
                'indice_fiscal': 'indice_fiscal',
                'area_do_terreno': 'area_terreno',
                'terreno': 'area_terreno',
                'vagas': 'vagas_garagem',
                'andar': 'andar',
                'pe_direito': 'pe_direito',
                'tipologia': 'tipologia'
            }
            df_global = df_global.rename(columns=mapeamento)
            
            features_obrigatorias = [
                'valor_total_declarado', 'area_privativa', 'indice_fiscal', 
                'area_terreno', 'vagas_garagem', 'andar', 'pe_direito', 'tipologia'
            ]
            
            for col in features_obrigatorias:
                if col not in df_global.columns:
                    df_global[col] = 0
                    st.warning(f"A coluna '{col}' não foi encontrada no arquivo. Preenchendo com valor 0.")

            st.success("✅ Base processada com segurança.")
            st.session_state.df_global = df_global
        except Exception as e:
            st.error(f"Erro crítico no processamento: {e}")
    else:
        if 'df_global' not in st.session_state:
            st.session_state.df_global = carregar_base_multitipologia_padrao()
            st.info("ℹ️ Utilizando base de dados padrão multi-tipologia integrada.")

    sub_casa, sub_apto, sub_lote, sub_galpao = st.tabs(["🏡 Casas", "🏢 Apartamentos", "📐 Lotes / Terrenos", "🏭 Galpões Comerciais"])
       
    with sub_casa:
        st.markdown("##### Parâmetros para Imóveis Horizontais")
        c1, c2, c3 = st.columns(3)
        area_casa = c1.number_input("Área Construída Privativa (m²)", min_value=10.0, value=120.0, key="c_a")
        terreno_casa = c1.number_input("Área Total do Terreno (m²)", min_value=10.0, value=360.0, key="c_t")
        quartos_casa = c2.slider("Quantidade de Quartos", 1, 6, 3, key="c_q")
        indice_casa = c2.number_input("Índice Fiscal da Quadra (Prefeitura)", min_value=0.0, value=1450.0, key="c_i")
        padrao_casa = c3.selectbox("Padrão Construtivo da Casa", ["Baixo", "Normal", "Alto"], index=1, key="c_p")
        if st.button("🚀 Calcular AVM de Casa"):
            st.session_state.alvo = {
                "tipologia": "CASA", "area": area_casa, "indice": indice_casa,
                "atributos": {"area_terreno": terreno_casa, "vagas_garagem": quartos_casa - 1, "andar": 0, "pe_direito": 3.0},
            }

    with sub_apto:
        st.markdown("##### Parâmetros para Edificações Verticais")
        a1, a2, a3 = st.columns(3)
        area_apto = a1.number_input("Área Privativa do Apartamento (m²)", min_value=10.0, value=75.0, key="ap_a")
        andar_apto = a1.number_input("Número do Andar / Pavimento", min_value=0, value=5, key="ap_an")
        vagas_apto = a2.slider("Vagas de Garagem no Subsolo", 0, 4, 1, key="ap_v")
        indice_apto = a2.number_input("Índice Fiscal da Quadra (Prefeitura)", min_value=0.0, value=2800.0, key="ap_i")
        padrao_apto = a3.selectbox("Padrão Construtivo do Prédio", ["Baixo", "Normal", "Alto"], index=1, key="ap_p")
        if st.button("🚀 Calcular AVM de Apartamento"):
            st.session_state.alvo = {
                "tipologia": "APARTAMENTO", "area": area_apto, "indice": indice_apto,
                "atributos": {"area_terreno": 0, "vagas_garagem": vagas_apto, "andar": andar_apto, "pe_direito": 2.8},
            }

    with sub_lote:
        st.markdown("##### Parâmetros para Solos Nus / Lotes")
        l1, l2 = st.columns(2)
        area_lote = l1.number_input("Área Total do Lote (m²)", min_value=10.0, value=450.0, key="lo_a")
        indice_lote = l2.number_input("Índice Fiscal da Quadra (Prefeitura)", min_value=0.0, value=1000.0, key="lo_i")
        topografia_lote = l1.selectbox("Topografia do Lote", ["Plano", "Aclive", "Declive"], key="lo_t")
        esquina_lote = l2.checkbox("Lote de Esquina", key="lo_e")
        if st.button("🚀 Calcular AVM de Lote"):
            st.session_state.alvo = {
                "tipologia": "LOTE", "area": area_lote, "indice": indice_lote,
                "atributos": {"area_terreno": area_lote, "vagas_garagem": 0, "andar": 0, "pe_direito": 0},
            }

    with sub_galpao:
        st.markdown("##### Parâmetros para Galpões Industriais / Logísticos")
        g1, g2 = st.columns(2)
        area_galpao = g1.number_input("Área Construída do Galpão (m²)", min_value=50.0, value=600.0, key="ga_a")
        terreno_galpao = g1.number_input("Área Total do Terreno (m²)", min_value=50.0, value=900.0, key="ga_t")
        pe_direito_galpao = g2.number_input("Pé-Direito Livre (m)", min_value=3.0, value=7.0, key="ga_pd")
        indice_galpao = g2.number_input("Índice Fiscal da Quadra (Prefeitura)", min_value=0.0, value=950.0, key="ga_i")
        if st.button("🚀 Calcular AVM de Galpão"):
            st.session_state.alvo = {
                "tipologia": "GALPAO", "area": area_galpao, "indice": indice_galpao,
                "atributos": {"area_terreno": terreno_galpao, "vagas_garagem": 0, "andar": 0, "pe_direito": pe_direito_galpao},
            }

    if 'alvo' in st.session_state:
        alvo = st.session_state.alvo
        st.write("---")
        st.subheader(f"📈 Resultado da Avaliação — {alvo['tipologia']}")

        df_atual = st.session_state.get('df_global', carregar_base_multitipologia_padrao())
        resultado, erro = executar_motor_ia(df_atual, alvo['tipologia'], alvo['area'], alvo['indice'], alvo['atributos'])

        if erro:
            st.error(f"⚠️ {erro}")
        else:
            v = resultado['valores']
            m1, m2, m3 = st.columns(3)
            m1.metric("Margem Mínima (Garantia)", f"R$ {v['v_min']:,.2f}")
            m2.metric("Valor de Face Estimado", f"R$ {v['v_medio']:,.2f}")
            m3.metric("Limite Máximo de Mercado", f"R$ {v['v_max']:,.2f}")
            st.caption(f"Precisão de Treino R² = {resultado['r2']} | Amostras utilizadas = {resultado['n_amostras']}")

            pdf_bytes = gerar_laudo_pdf_ia(
                tenant_selecionado, alvo['tipologia'], alvo['area'], v,
                resultado['r2'], resultado['n_amostras'],
                st.session_state.status_juridico_global,
                st.session_state.score_juridico_global,
            )
            st.download_button(
                "📄 Baixar Laudo AVM em PDF",
                data=pdf_bytes,
                file_name=f"laudo_avm_{alvo['tipologia'].lower()}.pdf",
                mime="application/pdf",
            )

with aba_juridico:
    st.subheader("Esteira de Risco Jurídico da Matrícula")
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
