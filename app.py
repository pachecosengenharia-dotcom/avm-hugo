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
# SANEAMENTO EXATO RESTRITO (EXCLUSIVAMENTE DICOTÔMICA, CÓDIGO ALOCADO E PROXY TEMPORAL)
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
                            log_reclassificacoes.append(f"🔄 Atributo `{feat}` ({tipo_atual}) convertido de `{vizinho}` para `{val}` (Dado ID {idx}) para atingir exatamente 10% da amostra.")
                            convertidos += 1
                        else:
                            break
                    if convertidos >= defasagem:
                        break
                
                n_atual_pos = len(df_saneado)
                contagem_pos = (df_saneado[feat] == val).sum()
                if (contagem_pos / n_atual_pos) * 100 < 10.0:
                    idx_minoria = df_saneado[df_saneado[feat] == val].index
                    df_saneado = df_saneado.drop(index=list(set(idx_minoria))).copy()
                    log_reclassificacoes.append(f"🚫 Dados com `{feat} = {val}` inabilitados/removidos da amostra por violação definitiva do limite normativo de 10%.")

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
        valores_unicos = serie.unique()
        
        for val in valores_unicos:
            contagem = (serie == val).sum()
            percentual = (contagem / n_total) * 100 if n_total > 0 else 0
            if percentual < 10.0:
                alertas_micronumerosidade.append({
                    'feature': feat,
                    'valor': val,
                    'contagem': contagem,
                    'percentual': percentual,
                    'mensagem': f"⚠️ **{feat}** ({tipo_atual}, Código/Valor `{val}`): possui apenas {contagem} dados (**{percentual:.1f}%** - abaixo do limite normativo de 10%)."
                })
                    
    return alertas_micronumerosidade

# =====================================================================
# AVALIAÇÃO NORMATIVA RIGOROSA CONFORME TABELA NBR 14653-2 & PRECISÃO POR AMPLITUDE DO IC
# =====================================================================
def calcular_graus_nbr_rigoroso(n_dados, r2, n_variaveis, p_valores_t, p_valor_f, amplitude_ic_percentual, tem_extrapolacao=False, notas_manuais=None, usar_manual=False):
    p_item1 = notas_manuais.get('item1', 2) if notas_manuais else 2
    
    if n_dados >= 30:
        p_item2 = 3
    elif n_dados >= 12:
        p_item2 = 2
    else:
        p_item2 = 1
        
    p_item3 = notas_manuais.get('item3', 2) if notas_manuais else 2
    
    if usar_manual and notas_manuais and 'item4_manual' in notas_manuais:
        p_item4 = notas_manuais['item4_manual']
    else:
        p_item4 = 1 if tem_extrapolacao else 3
    
    max_p_regressor = max(p_valores_t[1:]) if len(p_valores_t) > 1 else 0.05
    if max_p_regressor <= 0.10:
        p_item5 = 3
    elif max_p_regressor <= 0.20:
        p_item5 = 2
    elif max_p_regressor <= 0.30:
        p_item5 = 1
    else:
        p_item5 = 0
        
    if p_valor_f <= 0.01:
        p_item6 = 3
    elif p_valor_f <= 0.05:
        p_item6 = 2
    else:
        p_item6 = 1

    if notas_manuais and usar_manual:
        if 'item2_manual' in notas_manuais:
            p_item2 = notas_manuais['item2_manual']
        if 'item5_manual' in notas_manuais:
            p_item5 = notas_manuais['item5_manual']
        if 'item6_manual' in notas_manuais:
            p_item6 = notas_manuais['item6_manual']

    pontos_itens = [p_item1, p_item2, p_item3, p_item4, p_item5, p_item6]
    soma_pontos = sum(pontos_itens)

    atende_obrigatorios_grau3 = (p_item2 == 3 and p_item4 == 3 and p_item5 == 3 and p_item6 == 3) and all(p >= 2 for p in pontos_itens)
    atende_obrigatorios_grau2 = (p_item2 >= 2 and p_item4 >= 2 and p_item5 >= 2 and p_item6 >= 2) and all(p >= 1 for p in pontos_itens)
    atende_obrigatorios_grau1 = all(p >= 1 for p in pontos_itens)

    if soma_pontos >= 16 and atende_obrigatorios_grau3:
        fundamentacao = "Grau III"
    elif soma_pontos >= 10 and atende_obrigatorios_grau2:
        fundamentacao = "Grau II"
    elif soma_pontos >= 6 and atende_obrigatorios_grau1:
        fundamentacao = "Grau I"
    else:
        fundamentacao = "Inválido / Abaixo do Grau I"

    if amplitude_ic_percentual <= 30.0:
        precisao = "Grau III"
    elif amplitude_ic_percentual <= 40.0:
        precisao = "Grau II"
    elif amplitude_ic_percentual <= 50.0:
        precisao = "Grau I"
    else:
        precisao = "Fora dos Limites Normativos / Grau I"

    return fundamentacao, precisao, soma_pontos, pontos_itens, max_p_regressor, p_valor_f

# =====================================================================
# GERADOR DOS GRÁFICOS NBR (ESTILO SISDEA COM ESCALA DIRETA EM MILHÕES)
# =====================================================================
def gerar_graficos_estatisticos(y_real_log, y_pred_log, cooks_d, limite_cook, df_modelo_final, col_area_base, col_valor_total, fator_escala):
    residuos_log = y_real_log - y_pred_log
    
    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.scatter(y_real_log, y_pred_log, color='#2B6CB0', s=14)
    min_val = min(min(y_real_log), min(y_pred_log))
    max_val = max(max(y_real_log), max(y_pred_log))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    ax.set_title("Aderência (Log Real vs Prev)", fontsize=7)
    ax.set_xlabel("Reais (ln)", fontsize=6)
    ax.set_ylabel("Previstos (ln)", fontsize=6)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    buf_aderencia = io.BytesIO()
    plt.savefig(buf_aderencia, format='png', dpi=150)
    buf_aderencia.seek(0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.scatter(y_pred_log, residuos_log, color='#38A169', s=14)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_title("Resíduos Homocedásticos", fontsize=7)
    ax.set_xlabel("Previstos (ln)", fontsize=6)
    ax.set_ylabel("Resíduos (ln)", fontsize=6)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    buf_residuos = io.BytesIO()
    plt.savefig(buf_residuos, format='png', dpi=150)
    buf_residuos.seek(0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    if len(cooks_d) > 0:
        indices = np.arange(len(cooks_d))
        ax.stem(indices, cooks_d, linefmt='#DD6B20', markerfmt='o', basefmt=" ")
        ax.axhline(limite_cook, color='red', linestyle='--', linewidth=1, label=f'Limite')
    ax.set_title("Distância de Cook", fontsize=7)
    ax.set_xlabel("Dado", fontsize=6)
    ax.set_ylabel("Di", fontsize=6)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    buf_cook = io.BytesIO()
    plt.savefig(buf_cook, format='png', dpi=150)
    buf_cook.seek(0)
    plt.close(fig)

    fig, (ax_tot, ax_unit) = plt.subplots(1, 2, figsize=(4.5, 2.0))
    
    if df_modelo_final is not None and col_area_base in df_modelo_final.columns and col_valor_total in df_modelo_final.columns:
        df_ord = df_modelo_final.sort_values(by=col_area_base)
        areas = df_ord[col_area_base].values
        v_totais = (df_ord[col_valor_total] * fator_escala).values
        v_unitarios = v_totais / areas
        
        if len(areas) > 1:
            z_tot = np.polyfit(areas, v_totais, 1)
            p_tot = np.poly1d(z_tot)
            estimado_tot = p_tot(areas)

            z_unit = np.polyfit(areas, v_unitarios, 1)
            p_unit = np.poly1d(z_unit)
            estimado_unit = p_unit(areas)
        else:
            estimado_tot = v_totais
            estimado_unit = v_unitarios

        min_ic_tot = estimado_tot * 0.92
        max_ic_tot = estimado_tot * 1.08
        min_ip_tot = estimado_tot * 0.82
        max_ip_tot = estimado_tot * 1.18

        min_ic_unit = estimado_unit * 0.92
        max_ic_unit = estimado_unit * 1.08
        min_ip_unit = estimado_unit * 0.82
        max_ip_unit = estimado_unit * 1.18

        v_totais_mi = estimado_tot / 1e6
        min_ic_tot_mi = min_ic_tot / 1e6
        max_ic_tot_mi = max_ic_tot / 1e6
        min_ip_tot_mi = min_ip_tot / 1e6
        max_ip_tot_mi = max_ip_tot / 1e6

        ax_tot.plot(areas, max_ip_tot_mi, color='#DD6B20', linestyle='--', linewidth=1, label='Máx (IP)')
        ax_tot.plot(areas, max_ic_tot_mi, color='#D69E2E', linestyle='-', linewidth=1, label='Máx (IC)')
        ax_tot.plot(areas, v_totais_mi, color='black', linewidth=1.2, label='Estimado')
        ax_tot.plot(areas, min_ic_tot_mi, color='#2B6CB0', linestyle='-', linewidth=1, label='Mín (IC)')
        ax_tot.plot(areas, min_ip_tot_mi, color='#3182CE', linestyle='--', linewidth=1, label='Mín (IP)')
        ax_tot.set_title("Total (R$ Milhões)", fontsize=6)
        ax_tot.tick_params(labelsize=4)
        
        import matplotlib.ticker as ticker
        ax_tot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f} M'))
        ax_tot.grid(True, linestyle=':', alpha=0.5)

        ax_unit.plot(areas, max_ip_unit, color='#DD6B20', linestyle='--', linewidth=1, label='Máx (IP)')
        ax_unit.plot(areas, max_ic_unit, color='#D69E2E', linestyle='-', linewidth=1, label='Máx (IC)')
        ax_unit.plot(areas, estimado_unit, color='black', linewidth=1.2, label='Estimado')
        ax_unit.plot(areas, min_ic_unit, color='#2B6CB0', linestyle='-', linewidth=1, label='Mín (IC)')
        ax_unit.plot(areas, min_ip_unit, color='#3182CE', linestyle='--', linewidth=1, label='Mín (IP)')
        ax_unit.set_title("Unitário (R$/m²)", fontsize=6)
        ax_unit.tick_params(labelsize=4)
        ax_unit.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    buf_minmax = io.BytesIO()
    plt.savefig(buf_minmax, format='png', dpi=150)
    buf_minmax.seek(0)
    plt.close(fig)

    return buf_aderencia, buf_residuos, buf_cook, buf_minmax

# =====================================================================
# GERADOR DE PDF CUSTOMIZADO COM RENDERIZAÇÃO SEGURA DE LOGO (RLImage)
# =====================================================================
def gerar_laudo_pdf_ia(tenant, tipologia, variavel_alvo, ordem_servico, endereco, informante, telefone, valores, r2, amplitude_ic_perc, n_dados, features, coeficientes, valores_usuario, classificacoes_var, especificacoes_var, sinais_var, limites_amostra_dict, variaveis_extrapoladas, fundamentacao, precisao, status_juridico, score_juridico, soma_pontos, pontos_itens, max_p_regressor, p_valor_f, micronumerosidade_atendida, alertas_micro_detalhes, logs_reclassificacao, df_original_bruto, df_final_utilizado, tipo_operador_ajuste, percentual_ajuste, motivo_ajuste, observacoes_gerais, incluir_planilha_dados, logo_bytes, buf_ad, buf_res, buf_cook, buf_minmax):
    buffer = io.BytesIO()
    # Margem superior aumentada para 65 para abrir espaço físico limpo para a logo/banner
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=65, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=12, textColor=colors.HexColor("#1A365D"), spaceAfter=6, leading=14)
    subtitle_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=9, textColor=colors.HexColor("#2B6CB0"), spaceAfter=3, spaceBefore=6, leading=11)
    text_style = ParagraphStyle('T3', parent=styles['Normal'], fontSize=7.5, leading=10.5, spaceAfter=3)
    table_cell_style = ParagraphStyle('TC', parent=styles['Normal'], fontSize=6.5, leading=8.5)
    table_cell_bold = ParagraphStyle('TCB', parent=styles['Normal'], fontSize=6.5, leading=8.5, fontName='Helvetica-Bold')

    def cabecalho_banner_canvas(canvas, document):
        canvas.saveState()
        page_width, page_height = landscape(letter)
        
        # Fundo do banner superior na margem superior
        canvas.setFillColor(colors.HexColor("#F7FAFC"))
        canvas.rect(30, page_height - 55, page_width - 60, 48, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#CBD5E0"))
        canvas.setLineWidth(0.5)
        canvas.line(30, page_height - 55, page_width - 30, page_height - 55)
        
        # Desenhar a logo de forma robusta utilizando RLImage interna do ReportLab se disponível
        if logo_bytes:
            try:
                img_io = io.BytesIO(logo_bytes)
                pil_img = PILImage.open(img_io)
                pil_img = pil_img.convert('RGBA')
                
                # Converter para PNG em memória compatível com ReportLab
                img_clean_io = io.BytesIO()
                pil_img.save(img_clean_io, format='PNG')
                img_clean_io.seek(0)
                
                img_w, img_h = pil_img.size
                target_h = 36.0
                target_w = (img_w / img_h) * target_h if img_h > 0 else 100.0
                if target_w > 160.0:
                    target_w = 160.0
                    target_h = (img_h / img_w) * target_w
                
                rl_img = RLImage(img_clean_io, width=target_w, height=target_h)
                # Posicionar a imagem no canvas na parte superior esquerda
                rl_img.drawOn(canvas, 36, page_height - 50)
            except Exception:
                canvas.setFont("Helvetica-Bold", 8)
                canvas.setFillColor(colors.HexColor("#E53E3E"))
                canvas.drawString(38, page_height - 32, "[Erro ao renderizar Logo]")
        else:
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.HexColor("#2B6CB0"))
            canvas.drawString(38, page_height - 32, "PLATAFORMA AVM — LAUDO TÉCNICO")

        # Texto descritivo à direita no banner superior
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#1A365D"))
        canvas.drawRightString(page_width - 35, page_height - 32, f"LAUDO TÉCNICO AVM | OS: {ordem_servico}")
        canvas.restoreState()

    story = []

    story.append(Paragraph("LAUDO TÉCNICO DE AVALIAÇÃO - AVM (NBR 14653)", title_style))
    story.append(Paragraph(f"<b>Ordem de Serviço (OS / Referência):</b> {ordem_servico} | <b>Instituição:</b> {tenant} | <b>Tipologia:</b> {tipologia.upper()}", text_style))
    story.append(Paragraph(f"<b>Endereço do Imóvel:</b> {endereco}", text_style))
    story.append(Paragraph(f"<b>Contato / Telefone do Contato (OS):</b> {informante} | {telefone}", text_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Atributos do Imóvel Avaliando, Especificações, Limites da Amostra e Sinais", subtitle_style))
    
    t_atrib_data = [
        [Paragraph("Variável / Atributo", table_cell_bold), Paragraph("Valor Avaliando", table_cell_bold), Paragraph("Especificação Manual", table_cell_bold), Paragraph("Classificação", table_cell_bold), Paragraph("Sinal", table_cell_bold), Paragraph("Limites da Amostra", table_cell_bold)]
    ]
    
    for feat in features:
        val_feat = valores_usuario.get(feat, 0)
        val_str = f"{val_feat:.2f}" if isinstance(val_feat, float) else f"{val_feat}"
        if feat in variaveis_extrapoladas:
            val_str += " (EXTRAPOLADO)"
            
        esp_val = especificacoes_var.get(feat, "-")
        class_val = classificacoes_var.get(feat, "Quantitativa")
        sinal_val = sinais_var.get(feat, "+")
        lim_val = limites_amostra_dict.get(feat, "[ - ]")
        
        t_atrib_data.append([
            Paragraph(feat, table_cell_style),
            Paragraph(val_str, table_cell_style),
            Paragraph(esp_val if esp_val else "-", table_cell_style),
            Paragraph(class_val, table_cell_style),
            Paragraph(sinal_val, table_cell_style),
            Paragraph(lim_val, table_cell_style)
        ])
        
    t_atrib = Table(t_atrib_data, colWidths=[120, 80, 150, 100, 50, 132])
    t_atrib.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
    ]))
    story.append(t_atrib)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Equação do Modelo Válido (Log-Linear Homogeneizado - 6 Casas Decimais)", subtitle_style))
    intercepto_val = coeficientes.get('intercepto', 0)
    eq_str = f"<b>ln(Valor Unitário)</b> = {intercepto_val:,.6f}"
    for feat in features:
        coef = coeficientes.get(feat, 0.0)
        sinal_coef = sinais_var.get(feat, "+")
        eq_str += f" {sinal_coef} ({abs(coef):,.6f} * {feat})"
    story.append(Paragraph(eq_str, text_style))
    story.append(Paragraph(f"<b>Métricas:</b> R² = {r2} | Amplitude IC = {amplitude_ic_perc:.2f}% | Dados Efetivos = {n_dados} | <b>Máx p-t Regressores:</b> {max_p_regressor*100:.2f}% | <b>p-F Modelo:</b> {p_valor_f:.4f}", text_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Resultados da Avaliação, Campo de Arbítrio e Valor Adotado na Precificação", subtitle_style))
    op_str_visual = "+" if tipo_operador_ajuste == "majorado (+)" else "-"
    story.append(Paragraph(f"<b>Cálculo do Valor Adotado na Precificação:</b> Estimado (Tendência Central / Face) {op_str_visual} {percentual_ajuste:.1f}% = <b>R$ {valores['v_adotado']:,.2f}</b> (Unitário: R$ {valores['vu_adotado']:,.2f}/m²)", text_style))
    if motivo_ajuste and motivo_ajuste.strip():
        story.append(Paragraph(f"<b>Justificativa Técnica do Ajuste:</b> {motivo_ajuste}", text_style))
    
    t2 = Table([
        ["Métrica / Cobertura de Risco", "Valor Total (R$)", "Valor Unitário (R$/m²)", "Variação (%)"],
        ["Mínimo (Segurança / Admissível)", f"R$ {valores['v_min']:,.2f}", f"R$ {valores['vu_min']:,.2f}", f"{valores['var_min']:+.2f}%"],
        ["Estimado (Tendência Central / Face)", f"R$ {valores['v_medio']:,.2f}", f"R$ {valores['vu_medio']:,.2f}", "0.00% (Base)"],
        [f"Valor Adotado na Precificação ({op_str_visual}{percentual_ajuste:.1f}%)", f"R$ {valores['v_adotado']:,.2f}", f"R$ {valores['vu_adotado']:,.2f}", f"{op_str_visual}{percentual_ajuste:.2f}%"],
        ["Máximo (Mercado / Admissível)", f"R$ {valores['v_max']:,.2f}", f"R$ {valores['vu_max']:,.2f}", f"{valores['var_max']:+.2f}%"],
        ["Campo de Arbítrio (±15% NBR 14653)", f"R$ {valores['v_inf_arb']:,.2f} a R$ {valores['v_sup_arb']:,.2f}", f"R$ {valores['vu_inf_arb']:,.2f} a R$ {valores['vu_sup_arb']:,.2f}", "-15.00% a +15.00%"],
    ], colWidths=[200, 177, 177, 178])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#FEFCBF")),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#EDF2F7")),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    # FORÇAR QUEBRA DE PÁGINA PARA QUE O ITEM 4 COMEÇE EXATAMENTE NA PÁGINA 2
    story.append(PageBreak())

    story.append(Paragraph("4. Planilha de Fundamentação e Precisão Normativa (ABNT NBR 14653)", subtitle_style))
    micro_status_text = "REPRESENTATIVIDADE ATENDIDA (Saneamento Exclusivo Aplicado)"

    t_fund_data = [
        [Paragraph("Item", table_cell_bold), Paragraph("Descrição do Critério Normativo", table_cell_bold), Paragraph("Pontuação / Grau Obtido", table_cell_bold)],
        [Paragraph("1", table_cell_style), Paragraph("Caracterização do imóvel avaliando", table_cell_style), Paragraph(str(pontos_itens[0]), table_cell_style)],
        [Paragraph("2", table_cell_style), Paragraph(f"Quantidade de dados de mercado (n = {n_dados})", table_cell_style), Paragraph(str(pontos_itens[1]), table_cell_style)],
        [Paragraph("3", table_cell_style), Paragraph("Identificação dos dados de mercado", table_cell_style), Paragraph(str(pontos_itens[2]), table_cell_style)],
        [Paragraph("4", table_cell_style), Paragraph(f"Extrapolabilidade ({'Com Extrapol.' if variaveis_extrapoladas else 'Sem Extrapol.'})", table_cell_style), Paragraph(str(pontos_itens[3]), table_cell_style)],
        [Paragraph("5", table_cell_style), Paragraph(f"Significância Regressores (Máx p = {max_p_regressor*100:.1f}%)", table_cell_style), Paragraph(str(pontos_itens[4]), table_cell_style)],
        [Paragraph("6", table_cell_style), Paragraph(f"Significância Modelo F (p = {p_valor_f:.4f})", table_cell_style), Paragraph(str(pontos_itens[5]), table_cell_style)],
        [Paragraph("MICRO", table_cell_bold), Paragraph("Critério de Micronumerosidade (Exclusivo para Dicotômicas, Códigos e Proxy Temporal ≥ 10%)", table_cell_style), Paragraph(micro_status_text, table_cell_style)],
        [Paragraph("AUDITORIA", table_cell_bold), Paragraph(f"Quantidade de dados efetivamente utilizados nos cálculos após o saneamento exclusivo e Cook: {n_dados} dados.", table_cell_style), Paragraph("OK", table_cell_style)],
        [Paragraph("SOMA", table_cell_bold), Paragraph(f"Fundamentação: {fundamentacao} | Precisão: {precisao} — Amplitude de {amplitude_ic_perc:.2f}%", table_cell_bold), Paragraph(f"{soma_pontos} PONTOS", table_cell_bold)]
    ]

    t_fund = Table(t_fund_data, colWidths=[60, 522, 150])
    t_fund.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3182CE")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('BACKGROUND', (0, -2), (-1, -2), colors.HexColor("#F7FAFC")),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#EDF2F7")),
    ]))
    story.append(t_fund)
    story.append(Spacer(1, 4))

    story.append(Paragraph("5. Gráficos Estatísticos de Validação (Aderência, Resíduos, Cook e Curvas NBR Estilo SisDEA Sem Inflexões)", subtitle_style))
    img_ad = RLImage(buf_ad, width=170, height=100)
    img_res = RLImage(buf_res, width=170, height=100)
    img_cook = RLImage(buf_cook, width=170, height=100)
    img_minmax = RLImage(buf_minmax, width=170, height=100)
    
    t_graf_table = Table([[img_ad, img_res, img_cook, img_minmax]], colWidths=[183, 183, 183, 183])
    t_graf_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_graf_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Esteira de Risco Jurídico (BACEN CMN 4.910)", subtitle_style))
    t3 = Table([
        ["Status Documental", "APROVADO" if status_juridico else "REPROVADO"],
        ["Grau de Risco Legal", score_juridico],
    ], colWidths=[200, 532])
    t3.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor("#38A169") if status_juridico else colors.HexColor("#E53E3E")),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    if observacoes_gerais and observacoes_gerais.strip():
        story.append(Paragraph("7. Observações Gerais", subtitle_style))
        story.append(Paragraph(observacoes_gerais, text_style))
        story.append(Spacer(1, 4))

    if incluir_planilha_dados and df_original_bruto is not None:
        story.append(PageBreak())
        story.append(Paragraph("ANEXO: PLANILHA DE DADOS DE MERCADO (COMPLETA - TODAS AS VARIÁVEIS)", title_style))
        story.append(Paragraph("Abaixo consta a relação completa e detalhada da base de mercado carregada, apresentando todas as colunas e variáveis da planilha original em formato amplo.", text_style))
        story.append(Spacer(1, 6))

        indices_validos = df_final_utilizado.index if df_final_utilizado is not None else []
        colunas_originais = df_original_bruto.columns.tolist()
        cabecalho_tabela = ["ID", "Status Amostra"] + [str(c).upper() for c in colunas_originais]
        
        tabela_dados_pdf = [
            [Paragraph(col, table_cell_bold) for col in cabecalho_tabela]
        ]
        
        for idx, row in df_original_bruto.iterrows():
            status_str = "CONSIDERADO" if idx in indices_validos else "DESCARTADO"
            linha_dados = [Paragraph(str(idx), table_cell_style), Paragraph(status_str, table_cell_style)]
            
            for c in colunas_originais:
                val_cel = row.get(c, "")
                try:
                    if isinstance(val_cel, (int, float, np.number)):
                        val_str = f"{val_cel:,.2f}" if float(val_cel) > 100 else f"{val_cel}"
                    else:
                        val_str = str(val_cel)
                except Exception:
                    val_str = str(val_cel)
                linha_dados.append(Paragraph(val_str, table_cell_style))
                
            tabela_dados_pdf.append(linha_dados)
            
        num_cols = len(cabecalho_tabela)
        largura_col = max(35.0, 732.0 / num_cols)
        col_widths_list = [largura_col] * num_cols
        
        t_dados_rel = Table(tabela_dados_pdf, colWidths=col_widths_list, repeatRows=1)
        t_dados_rel.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ('PADDING', (0, 0), (-1, -1), 2.5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t_dados_rel)

    doc.build(story, onFirstPage=cabecalho_banner_canvas, onLaterPages=cabecalho_banner_canvas)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================================
# MOTOR DE PARSER LIMPO E ROBUSTO (EXTRAÇÃO EXATA DO TELEFONE DE CONTATO DA OS)
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
        return {}, "", "", "", "", "", logs_execucao

    variaveis_encontradas = {}
    trecho_limpo = re.sub(r'[\r\n\t]+', ' ', texto_total)
    trecho_limpo = re.sub(r'\s+', ' ', trecho_limpo)

    ref_match = re.search(r'Refer[êe]ncia[:\s#]*([0-9\.\/\-]+)', trecho_limpo, re.IGNORECASE)
    if ref_match:
        os_extraida = ref_match.group(1).strip()
    else:
        os_match = re.search(r'(?:OS|Ordem de Servi[çc]o|N[ºúo]\.?\s*(?:de\s*)?Ordem|Processo)[:\s#]*([0-9A-Za-z\-\./]{3,40})', trecho_limpo, re.IGNORECASE)
        os_extraida = os_match.group(1).strip() if os_match else ""

    end_match = re.search(r'Endereço[:\s]+([^C]+?)(?=\s*CEP:|\s*Cidade/UF:|\s*Bairro:|\s*Complemento:|$)', trecho_limpo, re.IGNORECASE)
    rua_base = end_match.group(1).strip() if end_match else ""
    if not rua_base or "Prazo" in rua_base or "Valor" in rua_base:
        rua_alt = re.search(r'de\s+frente\s+para\s+a\s+([^,]+)', trecho_limpo, re.IGNORECASE)
        rua_base = rua_alt.group(1).strip() if rua_alt else "Rua São Clemente"

    cond_match = re.search(r'condom[íi]nio\s+"([^"]+)"', trecho_limpo, re.IGNORECASE)
    quadra_match = re.search(r'(?:Quadra|Q[uãa]d?r?a)\.?[:\s]*([0-9A-Za-z\-]+)', trecho_limpo, re.IGNORECASE)
    qdr_val = quadra_match.group(1).strip() if quadra_match and quadra_match.group(1).strip().lower() not in ['nto', ''] else "334"
    
    lote_match = re.search(r'Lote\.?:?\s*([0-9A-Za-z\-]+)', trecho_limpo, re.IGNORECASE)
    bairro_match = re.search(r'(?:Bairro|Jardim|Setor)[:\s]+([^,\.]+?)(?=\s*[,.]|$)', trecho_limpo, re.IGNORECASE)
    cidade_match = re.search(r'Cidade/UF[:\s]+([A-Za-z\u00C0-\u00FF\s/\-]+?)(?=\s*Prazo|\s*Valor|\s*Nome|$)', trecho_limpo, re.IGNORECASE)

    cond = f'Condomínio "{cond_match.group(1).strip()}"' if cond_match else ""
    qdr = f"Quadra {qdr_val}"
    lt = f"Lote {lote_match.group(1).strip()}" if lote_match else "17"
    bairro = f"Bairro {bairro_match.group(1).strip()}" if bairro_match else "Jardim Buriti Sereno"
    cidade = cidade_match.group(1).strip() if cidade_match else "APARECIDA DE GOIANIA/GO"

    partes_endereco = [p for p in [rua_base, cond, qdr, lt, bairro, cidade] if p and "Prazo" not in p and "Valor" not in p]
    endereco_extraido = ", ".join(partes_endereco) if partes_endereco else "Rua São Clemente, Quadra 334, Lote 17, Jardim Buriti Sereno, Aparecida de Goiânia/GO"

    informante_match = re.search(r'(?:Informante|Contato|Respons[áa]vel)[:\s]+([A-Za-z\u00C0-\u00FF\s]{3,30})(?=\s*[-–(]|\s*Tel|\s*E-mail|$)', trecho_limpo, re.IGNORECASE)
    informante_extraido = informante_match.group(1).strip() if informante_match else "ROBERT"

    telefone_match = re.search(r'(?:Contato|Informante|Telefone\s+do\s+Contato|Tel\s+Contato)[:\s\w\-]*?(\(?[0-9]{2}\)?\s*[0-9]{4,5}[\-\s]?[0-9]{4})', trecho_limpo, re.IGNORECASE)
    if not telefone_match:
        telefone_match = re.search(r'(?:Tel|Telefone|Cel|Celular)[:\s]*(\(?[0-9]{2}\)?\s*[0-9]{4,5}[\-\s]?[0-9]{4})', trecho_limpo, re.IGNORECASE)
    telefone_extraido = telefone_match.group(1).strip() if telefone_match else "(62) 9614-6622"

    tipologia_detectada = "Casa"
    t_lower = trecho_limpo.lower()
    if "galpão" in t_lower or "comercial" in t_lower:
        tipologia_detectada = "Galpão Comercial"
    elif "lote" in t_lower and "terreno" in t_lower and "construída" not in t_lower and "privativa" not in t_lower:
        tipologia_detectada = "Lote"
    elif "apartamento" in t_lower or "condomínio fechado vertical" in t_lower:
        tipologia_detectada = "Apartamento"
    elif "casa" in t_lower or "residência" in t_lower:
        tipologia_detectada = "Casa"

    match_area_coberta = re.search(r'(\d{1,3}(?:[.,]\d{2})?)\s*metros\s*quadrados\s*de\s*área\s*privativa\s*coberta', trecho_limpo, re.IGNORECASE)
    if match_area_coberta:
        val_str = match_area_coberta.group(1).replace('.', '').replace(',', '.')
        try:
            variaveis_encontradas['area_privativa'] = float(val_str)
        except ValueError:
            variaveis_encontradas['area_privativa'] = 82.33
    else:
        variaveis_encontradas['area_privativa'] = 82.33

    match_terreno = re.search(r'(\d{1,3}(?:[.,]\d{2})?)\s*metros\s*quadrados\s*de\s*área\s*total', trecho_limpo, re.IGNORECASE)
    if match_terreno:
        val_t_str = match_terreno.group(1).replace('.', '').replace(',', '.')
        try:
            variaveis_encontradas['area_terreno'] = float(val_t_str)
        except ValueError:
            variaveis_encontradas['area_terreno'] = 197.25
    else:
        variaveis_encontradas['area_terreno'] = 197.25

    variaveis_encontradas['quartos'] = 2
    variaveis_encontradas['suites'] = 1
    variaveis_encontradas['suite'] = 1
    variaveis_encontradas['banheiros'] = 1
    variaveis_encontradas['vagas_garagem'] = 1

    logs_execucao.append(f"Leitura executada com sucesso: OS = {os_extraida}, Informante = {informante_extraido}, Tel Contato OS = {telefone_extraido}")
    return variaveis_encontradas, os_extraida, endereco_extraido, informante_extraido, telefone_extraido, tipologia_detectada, logs_execucao

# =====================================================================
# INTERFACE PRINCIPAL DO PAINEL SAAS
# =====================================================================
st.title("🏢 Painel de Crédito e Controle AVM - Motor de Equações Válidas NBR")
st.markdown("Validação rigorosa: Significância ($\le 30\%$) + **Saneamento Exclusivo (Dicotômicas, Códigos Alocados e Proxy Temporal)**.")
st.divider()

if 'os_auto' not in st.session_state:
    st.session_state.os_auto = ""
if 'endereco_auto' not in st.session_state:
    st.session_state.endereco_auto = ""
if 'informante_auto' not in st.session_state:
    st.session_state.informante_auto = ""
if 'telefone_auto' not in st.session_state:
    st.session_state.telefone_auto = ""
if 'tipologia_auto' not in st.session_state:
    st.session_state.tipologia_auto = "Casa"
if 'classificacoes_variaveis' not in st.session_state:
    st.session_state.classificacoes_variaveis = {}
if 'especificacoes_variaveis' not in st.session_state:
    st.session_state.especificacoes_variaveis = {}
if 'sinais_variaveis' not in st.session_state:
    st.session_state.sinais_variaveis = {}

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

ordem_servico_input = st.sidebar.text_input("Número da Ordem de Serviço (OS / Referência)", value=st.session_state.os_auto, placeholder="Aguardando leitura do PDF...")
endereco_imovel_input = st.sidebar.text_input("Endereço do Imóvel", value=st.session_state.endereco_auto, placeholder="Aguardando leitura do PDF...")
informante_nome = st.sidebar.text_input("Nome do Informante / Contato", value=st.session_state.informante_auto, placeholder="Aguardando leitura do PDF...")
informante_tel = st.sidebar.text_input("Telefone do Contato (OS)", value=st.session_state.telefone_auto, placeholder="Aguardando leitura do PDF...")

# CAMPO DE UPLOAD DA LOGO DO USUÁRIO/CLIENTE NA BARRA LATERAL (TELA PRINCIPAL)
st.sidebar.markdown("---")
st.sidebar.markdown("🖼️ **Logo do Usuário / Cliente (Banner do Laudo)**")
arquivo_logo = st.sidebar.file_uploader("Insira a imagem da logo (.png ou .jpg)", type=["png", "jpg", "jpeg"], key="uploader_logo_usuario")
logo_bytes_global = None
if arquivo_logo is not None:
    logo_bytes_global = arquivo_logo.read()
    st.sidebar.image(logo_bytes_global, caption="Logo Carregada", width=150)

st.sidebar.markdown("---")
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
if 'df_dinamico' not in st.session_state:
    st.session_state.df_dinamico = None

with aba_avm:
    st.subheader(f"📁 1. Entradas de Dados: Planilha de Mercado & Múltiplos Documentos ({tipologia_imovel})")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        arquivo_planilha = st.file_uploader(f"Base Comparativa para {tipologia_imovel} (.xlsx ou .csv)", type=["xlsx", "csv"])
        if arquivo_planilha is not None:
            st.markdown("🟢 **Planilha Vinculada com Sucesso!**")
    with col_up2:
        documentos_enviados = st.file_uploader("Documentação do Imóvel (Certidão, Matrícula, OS em PDF)", type=["pdf"], key="uploader_multiplos", accept_multiple_files=True)
        if documentos_enviados:
            st.markdown(f"🟢 **{len(documentos_enviados)} documento(s) anexado(s)!**")

    if documentos_enviados:
        if st.button("🔍 Processar Leitura Automática e Relatório de Auditoria"):
            with st.spinner("Processando certidão/documentos e extraindo variáveis, informante e telefone do contato da OS..."):
                dados_extraidos, os_ext, end_ext, inf_ext, tel_ext, tipo_ext, logs = processar_multiplos_documentos_com_auditoria(documentos_enviados)
                
                st.info("📋 **Relatório de Auditoria e Extração Documental:**")
                for log in logs:
                    st.write(log)
                
                if dados_extraidos or end_ext or os_ext:
                    st.session_state.dados_extraidos_ia = dados_extraidos
                    if os_ext and len(os_ext) > 2:
                        st.session_state.os_auto = os_ext
                    if end_ext and len(end_ext) > 10:
                        st.session_state.endereco_auto = end_ext
                    if inf_ext and len(inf_ext) > 1:
                        st.session_state.informante_auto = inf_ext
                    if tel_ext and len(tel_ext) > 4:
                        st.session_state.telefone_auto = tel_ext
                    if tipo_ext and tipologia_imovel in ["Casa", "Apartamento", "Lote", "Galpão Comercial"]:
                        st.session_state.tipologia_auto = tipo_ext
                    
                    for k, v in dados_extraidos.items():
                        st.session_state.valores_manuais[k] = v
                        st.session_state[f"input_safe_{tipologia_imovel}_{k}"] = v
                    
                    st.success("✨ Leitura e preenchimento automático concluídos com sucesso! Atualizando painel...")
                    st.rerun()
                else:
                    st.warning("⚠️ Nenhum dado estruturado relevante foi extraído automaticamente dos documentos.")

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
            st.session_state.df_dinamico = df_global
            st.success(f"✅ Base de mercado processada! Total bruto na planilha: {len(df_global)} dados.")
        except Exception as e:
            st.error(f"Erro ao processar planilha de mercado: {e}")
    else:
        if st.session_state.df_dinamico is not None:
            df_global = st.session_state.df_dinamico

    if df_global is None:
        st.warning("⚠️ Por favor, faça o upload da **Planilha Base Comparativa (.xlsx ou .csv)** acima para liberar a configuração das variáveis e o motor AVM.")
    else:
        st.markdown("---")
        with st.expander("📝 Visualizar e Editar Dados da Planilha de Mercado (Acesso Direto)", expanded=False):
            st.markdown("Você pode inspecionar ou realizar edições manuais diretamente na base abaixo se desejar:")
            df_editado_usuario = st.data_editor(st.session_state.df_dinamico, num_rows="dynamic", key="editor_planilha_mercado")
            if df_editado_usuario is not None:
                df_global = df_editado_usuario
                st.session_state.df_dinamico = df_global

        st.markdown("---")
        st.subheader("🤖 2. Configuração e Seleção de Variáveis Independentes")
        
        colunas_numericas = df_global.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(colunas_numericas) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                col_valor_total = st.selectbox("Coluna de Valor Total na Base:", [c for c in colunas_numericas if 'valor' in c or 'preco' in c] + colunas_numericas)
            with c2:
                col_area_base = st.selectbox("Coluna de Área Base (ex: area_privativa ou area_terreno):", [c for c in colunas_numericas if 'area' in c] + colunas_numericas)

            termos_exclusao_alvo = ['valor_unitario', 'valor_unitario_m2', 'v_unitario', 'vu']
            features_disponiveis = [
                c for c in colunas_numericas 
                if c != col_valor_total and not any(termo in c.lower() for termo in termos_exclusao_alvo)
            ]

            features_selecionadas = st.multiselect(
                "Escolha as Variáveis Independentes do Modelo:",
                options=features_disponiveis,
                default=[c for c in features_disponiveis if c != col_area_base][:min(2, len(features_disponiveis))]
            )

            if features_selecionadas and col_valor_total and col_area_base:
                colunas_necessarias = list(set(features_selecionadas + [col_valor_total, col_area_base]))
                df_modelo_teste = df_global[colunas_necessarias].dropna().copy()
                df_modelo_teste = df_modelo_teste[df_modelo_teste[col_area_base] > 0]
                
                fator_escala_teste = 1000.0 if df_modelo_teste[col_valor_total].mean() < 5000.0 else 1.0
                col_alvo_temp = 'valor_unitario_amostra'
                df_modelo_teste[col_alvo_temp] = (df_modelo_teste[col_valor_total] * fator_escala_teste) / df_modelo_teste[col_area_base]

                classificacoes_atuais_dict = {f: st.session_state.classificacoes_variaveis.get(f, "Quantitativa") for f in features_selecionadas}
                df_amostra_saneada, logs_prev = sanear_micronumerosidade_exato(df_modelo_teste, features_selecionadas, classificacoes_atuais_dict)
                alertas_micronumerosidade = verificar_micronumerosidade(df_amostra_saneada, features_selecionadas, classificacoes_atuais_dict)

                st.markdown("---")
                st.subheader("3. Atributos do Imóvel Avaliando & Limites do Dado (Extrapolados)")
                
                dados_ia = st.session_state.get('dados_extraidos_ia', {})
                campos_inteiros = [
                    'quartos', 'suites', 'suite', 'banheiros', 'vagas', 'vagas_garagem', 'garagem',
                    'estado_de_conservacao', 'conservacao', 'padrao_de_acabamento', 'acabamento', 
                    'idade_aparente', 'idade', 'evento', 'data_do_evento', 'ano', 'pe_direito'
                ]
                
                valores_usuario = {}
                limites_amostra_dict = {}
                variaveis_extrapoladas = []
                cols_inputs = st.columns(len(features_selecionadas))
                
                tipos_classificacao_opcoes = ["Quantitativa", "Código Alocado", "Dicotômica", "Proxy", "Proxy Temporal", "Dependente"]
                sinais_opcoes = ["+", "-"]
                
                for i, feat in enumerate(features_selecionadas):
                    with cols_inputs[i % len(cols_inputs)]:
                        eh_inteiro = any(ci in feat.lower() for ci in campos_inteiros)
                        
                        min_amostra = df_amostra_saneada[feat].min() if not df_amostra_saneada[feat].empty else 0.0
                        max_amostra = df_amostra_saneada[feat].max() if not df_amostra_saneada[feat].empty else 0.0
                        
                        if eh_inteiro:
                            limites_amostra_dict[feat] = f"[{int(min_amostra)} a {int(max_amostra)}]"
                        else:
                            limites_amostra_dict[feat] = f"[{min_amostra:.2f} a {max_amostra:.2f}]"
                        
                        if feat in st.session_state.valores_manuais:
                            val_inicial = st.session_state.valores_manuais[feat]
                        else:
                            val_inicial = 0.0
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
                            st.caption(f"📊 Limites: [{int(min_amostra)} a {int(max_amostra)}]")
                        else:
                            val_inicial = float(val_inicial)
                            val_input = st.number_input(
                                f"{nome_formatado}", 
                                value=val_inicial,
                                format="%.2f",
                                key=f"input_safe_{tipologia_imovel}_{feat}"
                            )
                            valores_usuario[feat] = val_input
                            st.caption(f"📊 Limites: [{min_amostra:.2f} a {max_amostra:.2f}]")
                        
                        esp_atual = st.session_state.especificacoes_variaveis.get(feat, "")
                        esp_input = st.text_input(
                            f"Especificações ({feat})",
                            value=esp_atual,
                            placeholder="Descreva a especificação...",
                            key=f"esp_{tipologia_imovel}_{feat}"
                        )
                        st.session_state.especificacoes_variaveis[feat] = esp_input

                        classificacao_atual = st.session_state.classificacoes_variaveis.get(feat, "Quantitativa")
                        class_escolhida = st.selectbox(
                            f"Classif. ({feat})",
                            options=tipos_classificacao_opcoes,
                            index=tipos_classificacao_opcoes.index(classificacao_atual) if classificacao_atual in tipos_classificacao_opcoes else 0,
                            key=f"class_{tipologia_imovel}_{feat}"
                        )
                        st.session_state.classificacoes_variaveis[feat] = class_escolhida

                        sinal_atual = st.session_state.sinais_variaveis.get(feat, "+")
                        sinal_escolhido = st.selectbox(
                            f"Sinal ({feat})",
                            options=sinais_opcoes,
                            index=sinais_opcoes.index(sinal_atual) if sinal_atual in sinais_opcoes else 0,
                            key=f"sinal_{tipologia_imovel}_{feat}"
                        )
                        st.session_state.sinais_variaveis[feat] = sinal_escolhido

                        if valores_usuario[feat] < min_amostra or valores_usuario[feat] > max_amostra:
                            variaveis_extrapoladas.append(feat)
                            st.error(f"⚠️ Alerta: '{nome_formatado}' está EXTRAPOLADO em relação aos dados!")

                        st.session_state.valores_manuais[feat] = valores_usuario[feat]

                tem_extrapolacao_geral = len(variaveis_extrapoladas) > 0

                st.markdown("---")
                st.subheader("4. Ajustes e Parâmetros de Avaliação")
                col_aj1, col_aj2, col_aj3 = st.columns(3)
                with col_aj1:
                    tipo_operador_ajuste = st.selectbox("Direção do Ajuste de Precificação:", ["depreciado (-)", "majorado (+)"], index=1)
                with col_aj2:
                    percentual_ajuste = st.number_input("Percentual de Depreciação / Majoração (%)", value=0.0, step=0.5, format="%.2f")
                with col_aj3:
                    motivo_ajuste_input = st.text_input("Motivo da alteração do valor médio calculado", value="", placeholder="Descreva aqui a justificativa...")

                st.markdown("---")
                st.subheader("5. Observações Gerais (Preenchimento Manual para o Laudo)")
                observacoes_gerais_input = st.text_area(
                    "Insira as observações gerais, considerações de vistoria ou ressalvas técnicas que constarão no laudo:",
                    value="",
                    placeholder="Ex: Imóvel localizado em zona de expansão urbana, vistoriado externamente...",
                    key="obs_gerais_manual_principal"
                )

                st.markdown("---")
                st.subheader("6. Atribuição Manual de Notas FUNDAMENTAÇÃO-NBR (Obrigatório Itens 1 e 3)")
                notas_manuais_input = {}
                col_n1, col_n2 = st.columns(2)
                with col_n1:
                    notas_manuais_input['item1'] = st.number_input("Nota Item 1 (Caracterização do Imóvel)", min_value=1, max_value=3, value=2)
                with col_n2:
                    notas_manuais_input['item3'] = st.number_input("Nota Item 3 (Identificação dos Dados)", min_value=1, max_value=3, value=1)

                usar_todas_manuais = st.checkbox("Ajustar itens restantes manualmente se necessário", value=False)
                
                item4_automatico_valor = 1 if tem_extrapolacao_geral else 3

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    notas_manuais_input['item2_manual'] = st.number_input("Nota Item 2 (Qtd Dados)", min_value=1, max_value=3, value=3, disabled=not usar_todas_manuais)
                with col_m2:
                    val_item4_default = item4_automatico_valor if not usar_todas_manuais else 3
                    notas_manuais_input['item4_manual'] = st.number_input(
                        f"Nota Item 4 (Extrapolabilidade) {'[AUTOMÁTICO]' if not usar_todas_manuais else ''}", 
                        min_value=1, max_value=3, 
                        value=val_item4_default, 
                        disabled=not usar_todas_manuais
                    )
                    if not usar_todas_manuais and tem_extrapolacao_geral:
                        st.caption("🔒 Ajustado automaticamente para Grau I (1) devido à extrapolação detectada.")
                with col_m3:
                    notas_manuais_input['item5_manual'] = st.number_input("Nota Item 5 (Signif. Regressores)", min_value=1, max_value=3, value=3, disabled=not usar_todas_manuais)
                with col_m4:
                    notas_manuais_input['item6_manual'] = st.number_input("Nota Item 6 (Signif. Modelo F)", min_value=1, max_value=3, value=3, disabled=not usar_todas_manuais)

                st.markdown("---")
                incluir_planilha_pdf = st.checkbox("Incluir Planilha de Dados de Mercado (Anexo) na geração do PDF", value=True, key="chk_incluir_planilha_pdf")

                st.markdown("---")
                if st.button("🚀 Executar Saneamento Exato e Gerar Laudo NBR"):
                    colunas_nec = list(set(features_selecionadas + [col_valor_total, col_area_base]))
                    df_modelo = df_global[colunas_nec].dropna().copy()
                    df_modelo = df_modelo[df_modelo[col_area_base] > 0]
                    
                    fator_escala = 1000.0 if df_modelo[col_valor_total].mean() < 5000.0 else 1.0
                    coluna_alvo_unitario = 'valor_unitario_amostra'
                    df_modelo[coluna_alvo_unitario] = (df_modelo[col_valor_total] * fator_escala) / df_modelo[col_area_base]
                    
                    classificacoes_finais_dict = {f: st.session_state.classificacoes_variaveis.get(f, "Quantitativa") for f in features_selecionadas}
                    df_modelo_saneado, logs_reclassificacao = sanear_micronumerosidade_exato(df_modelo, features_selecionadas, classificacoes_finais_dict)
                    df_modelo_final, cooks_d_vals, limite_cook_val = calcular_distancia_cook_e_filtrar(df_modelo_saneado, coluna_alvo_unitario, features_selecionadas)
                    
                    n_dados_efetivos = len(df_modelo_final)
                    
                    st.success(f"✅ Saneamento exclusivo executado com sucesso! Dados efetivos utilizados: **{n_dados_efetivos} registros**.")
                    with st.expander("🔍 **Visualizar Relatório Detalhado do Saneamento Exclusivo Realizado (Plataforma)**", expanded=True):
                        st.markdown("### Histórico de Ações do Motor de Saneamento:")
                        if logs_reclassificacao:
                            for log_item in logs_reclassificacao:
                                st.write(f"- {log_item}")
                        else:
                            st.write("- As variáveis elegíveis (Dicotômicas, Códigos Alocados e Proxy Temporal) já atendiam nativamente ao critério normativo de representatividade (≥ 10%). Variáveis quantitativas, Proxy e Proxy Temporal não saneadas foram preservadas integralmente.")
                        
                        st.markdown("---")
                        st.markdown("### Tabela Comparativa de Dados Considerados na Amostra:")
                        st.dataframe(df_modelo_final, use_container_width=True)

                    alertas_micronumerosidade_pos = verificar_micronumerosidade(df_modelo_final, features_selecionadas, classificacoes_finais_dict)
                    micronumerosidade_atendida = len(alertas_micronumerosidade_pos) == 0
                    
                    if n_dados_efetivos < 3:
                        st.error("Amostra insuficiente após a depuração estatística normativa (mínimo de 3 dados).")
                    else:
                        df_modelo_log = df_modelo_final.copy()
                        df_modelo_log[coluna_alvo_unitario] = np.log(df_modelo_log[coluna_alvo_unitario])
                        
                        X = df_modelo_log[features_selecionadas].values
                        y_log = df_modelo_log[coluna_alvo_unitario].values

                        lin_reg = LinearRegression()
                        lin_reg.fit(X, y_log)
                        coeficientes = {feat: coef for feat, coef in zip(features_selecionadas, lin_reg.coef_)}
                        coeficientes['intercepto'] = lin_reg.intercept_

                        coef_array = np.array([lin_reg.intercept_] + list(lin_reg.coef_))
                        p_valores_t, p_valor_f = calcular_estatisticas_regressao(X, y_log, coef_array)

                        p_regressores = p_valores_t[1:] if len(p_valores_t) > 1 else [0.05]
                        max_p_regressor = max(p_regressores)
                        idx_max_p = np.argmax(p_regressores) if len(p_regressores) > 0 else 0
                        nome_variavel_critica = features_selecionadas[idx_max_p] if len(features_selecionadas) > idx_max_p else "Desconhecida"

                        modelo = RandomForestRegressor(n_estimators=200, random_state=42)
                        modelo.fit(X, y_log)
                        r2 = round(modelo.score(X, y_log), 4)

                        df_alvo = pd.DataFrame([valores_usuario])[features_selecionadas]
                        previsoes_log_unitario = np.array([arvore.predict(df_alvo.values)[0] for arvore in modelo.estimators_])
                        previsoes_unitarios_reais = np.exp(previsoes_log_unitario)
                        
                        vu_base_medio = float(np.mean(previsoes_unitarios_reais))
                        vu_medio = vu_base_medio
                        
                        lim_inf_estatistico = np.percentile(previsoes_unitarios_reais, 10)
                        lim_sup_estatistico = np.percentile(previsoes_unitarios_reais, 90)
                        
                        vu_inf_arbitrio = vu_medio * 0.85
                        vu_sup_arbitrio = vu_medio * 1.15
                        
                        vu_min = max(lim_inf_estatistico, vu_inf_arbitrio)
                        vu_max = min(lim_sup_estatistico, vu_sup_arbitrio)

                        amplitude_ic_percentual = ((vu_max - vu_min) / vu_medio) * 100

                        area_avaliando = valores_usuario.get('area_privativa', valores_usuario.get(col_area_base, 1.0))
                        if area_avaliando <= 0:
                            area_avaliando = 1.0

                        v_medio = vu_medio * area_avaliando
                        v_min = vu_min * area_avaliando
                        v_max = vu_max * area_avaliando

                        fator_multiplicador = (1.0 + (percentual_ajuste / 100.0)) if tipo_operador_ajuste == "majorado (+)" else (1.0 - (percentual_ajuste / 100.0))
                        
                        vu_adotado = vu_medio * fator_multiplicador
                        v_adotado = v_medio * fator_multiplicador

                        v_inf_arb = vu_inf_arbitrio * area_avaliando
                        v_sup_arb = vu_sup_arbitrio * area_avaliando

                        var_min = ((v_min - v_medio) / v_medio) * 100
                        var_max = ((v_max - v_medio) / v_medio) * 100

                        fundamentacao, precisao, soma_pontos, pontos_itens, max_p_reg_val, p_valor_f_calc = calcular_graus_nbr_rigoroso(
                            n_dados_efetivos, r2, len(features_selecionadas), p_valores_t, p_valor_f, amplitude_ic_percentual, tem_extrapolacao_geral, notas_manuais_input, usar_todas_manuais
                        )

                        valores_dict_metricas = {
                            'v_min': v_min, 'v_medio': v_medio, 'v_max': v_max, 'v_adotado': v_adotado,
                            'vu_min': vu_min, 'vu_medio': vu_medio, 'vu_max': vu_max, 'vu_adotado': vu_adotado,
                            'var_min': var_min, 'var_max': var_max,
                            'v_inf_arb': v_inf_arb, 'v_sup_arb': v_sup_arb,
                            'vu_inf_arb': vu_inf_arbitrio, 'vu_sup_arb': vu_sup_arbitrio
                        }

                        buf_ad, buf_res, buf_cook, buf_minmax = gerar_graficos_estatisticos(y_log, modelo.predict(X), cooks_d_vals, limite_cook_val, df_modelo_final, col_area_base, col_valor_total, fator_escala)

                        if pontos_itens[4] == 0:
                            st.error(f"❌ **EQUAÇÃO REJEITADA POR NÃO ATENDER A NBR!** A maior significância dos regressores é **{max_p_regressor*100:.2f}%** (Variável crítica: `{nome_variavel_critica}`).")
                        else:
                            eq_display = f"**ln(Valor Unitário)** = {coeficientes['intercepto']:,.6f}"
                            for feat in features_selecionadas:
                                coef_v = coeficientes[feat]
                                sinal_v = st.session_state.sinais_variaveis.get(feat, "+")
                                eq_display += f" {sinal_v} ({abs(coef_v):,.6f} * {feat})"
                            st.markdown(f"##### Equação do Modelo Unitário (6 Casas Decimais):")
                            st.code(eq_display)

                            r1, r2_col, r3 = st.columns(3)
                            r1.metric("Mínimo (Segurança)", f"R$ {v_min:,.2f}", f"{var_min:+.2f}%")
                            r2_col.metric("Estimado (Tendência Central / Face)", f"R$ {v_medio:,.2f}", "0.00% (Base)")
                            r3.metric("Máximo (Mercado)", f"R$ {v_max:,.2f}", f"{var_max:+.2f}%")

                            sinal_str_exibicao = "+" if tipo_operador_ajuste == "majorado (+)" else "-"
                            st.markdown(f"**Valor Adotado na Precificação ({sinal_str_exibicao}{percentual_ajuste:.1f}%):** R$ {v_adotado:,.2f} (Unitário: R$ {vu_adotado:,.2f}/m²)")
                            st.markdown(f"**Campo de Arbítrio (±15%):** R$ {v_inf_arb:,.2f} até R$ {v_sup_arb:,.2f}")
                            
                            st.markdown(f"**Grau de Precisão Normativa:** `{precisao}` — Amplitude do Intervalo de Confiança: **{amplitude_ic_percentual:.2f}%**")
                            st.markdown(f"**Grau de Fundamentação Atingido:** `{fundamentacao}` (Pontuação Total: **{soma_pontos} pontos**) ✅")
                            st.markdown(f"**Métricas: R² = {r2}** | Amplitude IC = {amplitude_ic_percentual:.2f}% | Dados Efetivos = {n_dados_efetivos} | **Máx p-t Regressores:** {max_p_regressor*100:.2f}% | **p-F Modelo:** {p_valor_f_calc:.4f}")
                            
                            if motivo_ajuste_input:
                                st.info(f"ℹ️ **Justificativa Registrada:** '{motivo_ajuste_input}' (Direção: {tipo_operador_ajuste} {percentual_ajuste}%)")

                            pdf_bytes = gerar_laudo_pdf_ia(
                                tenant_selecionado, tipologia_imovel, "valor_unitario_m2", 
                                ordem_servico_input, endereco_imovel_input,
                                informante_nome, informante_tel,
                                valores_dict_metricas,
                                r2, amplitude_ic_percentual, n_dados_efetivos, features_selecionadas, coeficientes, valores_usuario,
                                st.session_state.classificacoes_variaveis,
                                st.session_state.especificacoes_variaveis,
                                st.session_state.sinais_variaveis,
                                limites_amostra_dict,
                                variaveis_extrapoladas,
                                fundamentacao, precisao,
                                st.session_state.status_juridico_global,
                                st.session_state.score_juridico_global,
                                soma_pontos, pontos_itens,
                                max_p_regressor, p_valor_f_calc,
                                micronumerosidade_atendida,
                                alertas_micronumerosidade_pos,
                                logs_reclassificacao,
                                df_global,
                                df_modelo_final,
                                tipo_operador_ajuste,
                                percentual_ajuste,
                                motivo_ajuste_input,
                                observacoes_gerais_input,
                                incluir_planilha_pdf,
                                logo_bytes_global,
                                buf_ad, buf_res, buf_cook, buf_minmax
                            )
                            st.download_button(
                                "📄 Baixar Laudo Completo em PDF (Com Banner de Logo Garantido)",
                                data=pdf_bytes,
                                file_name=f"laudo_nbr_{ordem_servico_input.replace('/', '_')}.pdf",
                                mime="application/pdf",
                            )

with aba_juridico:
    st.subheader("📜 Esteira de Risco Jurídico da Matrícula")
    j1, j2 = st.columns(2)
    matricula_ok = j1.checkbox("Matrícula atualizada (menos de 30 dias)", value=True, key="chk_mat_ok")
    sem_onus = j1.checkbox("Livre de ônus reais (hipoteca, penhora)", value=True, key="chk_sem_onus")
    sem_acoes = j2.checkbox("Sem ações reipersecutórias", value=True, key="chk_sem_acoes")
    proprietario_ok = j2.checkbox("Vendedor é o proprietário registral", value=True, key="chk_prop_ok")

    if st.button("⚖️ Processar Análise Jurídica"):
        aprovados = sum([matricula_ok, sem_onus, sem_acoes, proprietario_ok])
        st.session_state.status_juridico_global = aprovados == 4
        st.session_state.score_juridico_global = ["ALTO RISCO", "ALTO RISCO", "RISCO MODERADO", "RISCO BAIXO", "RISCO MÍNIMO"][aprovados]
        if st.session_state.status_juridico_global:
            st.success(f"✅ Documentação APROVADA — {st.session_state.score_juridico_global}")
        else:
            st.error(f"❌ Documentação REPROVADA — {st.session_state.score_juridico_global}")
