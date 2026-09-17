"""
Pipeline de Agrupamento (Clustering) de Nódulos de Tireoide — ERAMIA 2026
Artigo: "Análise do Comportamento de Nódulos de Tireoide com Citologia Indeterminada por Meio de Algoritmos de Agrupamento de Dados"
Autores: Mariana Luísa Gonçalves, Ana Trindade Winck, Luciano Costa Blomberg (UFCSPA)
Conferência: Escola Regional de Aprendizado de Máquina e Inteligência Artificial do RS (ERAMIA 2026 - SBC)
https://eramia-rs.sbc.org.br/2026/#/

Método Principal:
  - Matriz de Dissimilaridade de Gower (para dados clínicos mistos).
  - K-Medoids (PAM - Partitioning Around Medoids).
  - Validação Interna com Coeficiente de Silhueta.
  - Agrupamento Hierárquico Aglomerativo (Dendrograma).
  - Projeção 2D com PCA e MDS.
  - Caracterização clínica dos clusters e validação externa de malignidade.
  - Versionamento automático: cada execução cria uma subpasta com data e hora.
"""

import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import gower

# Configuração estética dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Diretório base de resultados
DIR_RESULTADOS = "resultados"
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# ==============================================================================
# 1. CARGA E SEPARAÇÃO DAS VARIÁVEIS
# ==============================================================================
def carregar_e_preparar_dados(caminho_csv):
    """
    Carrega a base de nódulos e separa as variáveis em:
    - Identificadores
    - Variáveis de Agrupamento (Features clínicas e ultrassonográficas)
    - Variáveis Externas de Desfecho (para validação clínica posterior)
    """
    print(f"\n[1] Lendo base de dados: {caminho_csv}")
    df = pd.read_csv(caminho_csv, sep=";", encoding="utf-8")
    
    # Identificadores
    colunas_id = ['id_paciente', 'id_nodulo']
    
    # Variáveis clínico-ultrassonográficas de agrupamento (conforme metodologia do artigo ERAMIA 2026)
    colunas_features = [
        'idade',
        'sexo',
        'citologia',
        'usg_tamanho_nodulo_maior',
        'usg_conteudo_nodulo',
        'usg_ecogenicidade_nodulo',
        'usg_contorno_nodulo',
        'usg_formato',
        'usg_halo_nodulo',
        'usg_fluxo_nodulo',
        'usg_microcalcificacoes',
        'usg_linfonodos',
        'usg_acr_nodulo'
    ]
    
    # Variáveis externas de validação clínica (NÃO entram no clustering)
    colunas_desfecho = ['cancer_de_tireoide', 'diagnostico_definitivo']
    
    df_features = df[colunas_features].copy()
    
    # Especificação das colunas categóricas para o cálculo da distância de Gower
    is_categorical = [col not in ['idade', 'usg_tamanho_nodulo_maior'] for col in colunas_features]
    
    print(f"    Total de instâncias carregadas: {len(df)}")
    print(f"    Variáveis de agrupamento selecionadas ({len(colunas_features)}): {colunas_features}")
    return df, df_features, is_categorical, colunas_features, colunas_desfecho

# ==============================================================================
# 2. CÁLCULO DA MATRIZ DE DISSIMILARIDADE DE GOWER
# ==============================================================================
def calcular_matriz_gower(df_features, is_categorical):
    """
    Calcula a matriz de dissimilaridade de Gower para integrar
    variáveis contínuas, ordinais e binárias/categóricas.
    """
    print("\n[2] Calculando Matriz de Dissimilaridade de Gower...")
    dist_matrix = gower.gower_matrix(df_features, cat_features=is_categorical)
    # Garante simetria e diagonal zerada por precisão numérica
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    print(f"    Matriz gerada com formato: {dist_matrix.shape} (distâncias entre 0.0 e 1.0)")
    return dist_matrix

# ==============================================================================
# 3. DETERMINAÇÃO DO NÚMERO ÓTIMO DE CLUSTERS (COEFICIENTE DE SILHUETA)
# ==============================================================================
def avaliar_numero_clusters(dist_matrix, dir_saida, k_min=2, k_max=6):
    """
    Avalia a qualidade dos agrupamentos para k de 2 a 6 utilizando o Coeficiente de Silhueta.
    Gera gráfico com linha estilizada no diretório de saída e espelha em resultados/.
    """
    print(f"\n[3] Avaliando número de clusters (k={k_min} até k={k_max})...")
    k_range = list(range(k_min, k_max + 1))
    silhouette_scores = []
    
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42, method='pam')
        labels = kmedoids.fit_predict(dist_matrix)
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)
        print(f"    k = {k}: Silhueta Média = {score:.4f}")
        
    melhor_k = k_range[np.argmax(silhouette_scores)]
    melhor_score = max(silhouette_scores)
    print(f"    -> Melhor k sugerido pela silhueta: k={melhor_k} (Score: {melhor_score:.4f})")
    
    # Plotagem da Curva de Silhueta (estilo idêntico ao artigo)
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.plot(k_range, silhouette_scores, marker='o', linewidth=2.5, color='#c04500', markersize=8)
    plt.axvline(x=melhor_k, color='dimgray', linestyle='--', linewidth=1.5, label=f'k Ótimo ({melhor_k})')
    plt.title('Coeficiente de Silhueta Médio por k (K-Medoids)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Número de Clusters (k)', fontsize=10)
    plt.ylabel('Score de Silhueta', fontsize=10)
    plt.xticks(k_range)
    plt.tight_layout()
    
    nome_fig = "curva_silhueta_kmedoids.png"
    caminho_exec = os.path.join(dir_saida, nome_fig)
    caminho_raiz = os.path.join(DIR_RESULTADOS, nome_fig)
    plt.savefig(caminho_exec)
    plt.savefig(caminho_raiz)
    plt.close()
    print(f"    Gráfico salvo em: {caminho_exec}")
    
    return melhor_k

# ==============================================================================
# 4. EXECUÇÃO DO K-MEDOIDS (ALGORITMO PRINCIPAL)
# ==============================================================================
def executar_kmedoids(dist_matrix, n_clusters=3):
    """
    Ajusta o algoritmo K-Medoids (PAM) usando a matriz pré-computada de Gower.
    Retorna os rótulos e os índices dos medoides reais.
    """
    print(f"\n[4] Ajustando K-Medoids com k={n_clusters}...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42, method='pam')
    labels = kmedoids.fit_predict(dist_matrix)
    medoid_indices = kmedoids.medoid_indices_
    
    score_final = silhouette_score(dist_matrix, labels, metric='precomputed')
    print(f"    Modelo convergido com sucesso.")
    print(f"    Coeficiente de Silhueta Final: {score_final:.4f}")
    print(f"    Índices dos Medoides Reais na base: {medoid_indices.tolist()}")
    return labels, medoid_indices, score_final

# ==============================================================================
# 5. EXECUÇÃO DO AGRUPAMENTO HIERÁRQUICO AGLOMERATIVO (COMPLEMENTAR)
# ==============================================================================
def executar_hierarquico(dist_matrix, dir_saida, n_clusters=3):
    """
    Gera o dendrograma hierárquico a partir da matriz de Gower (Average Linkage)
    e computa os clusters hierárquicos para comparação.
    """
    print(f"\n[5] Gerando Agrupamento Hierárquico Aglomerativo (Average Linkage)...")
    # Converte matriz condensada para linkage
    dist_condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(dist_condensed, method='average')
    
    # Plotagem do Dendrograma
    plt.figure(figsize=(9, 4.8), dpi=300)
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=9, show_contracted=True)
    plt.title('Dendrograma do Agrupamento Hierárquico (Matriz de Gower)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Nódulos Agrupados (Subárvores resumidas)', fontsize=10)
    plt.ylabel('Distância de Dissimilaridade (Gower)', fontsize=10)
    plt.tight_layout()
    
    nome_fig = "dendrograma_hierarquico.png"
    caminho_exec = os.path.join(dir_saida, nome_fig)
    caminho_raiz = os.path.join(DIR_RESULTADOS, nome_fig)
    plt.savefig(caminho_exec)
    plt.savefig(caminho_raiz)
    plt.close()
    print(f"    Dendrograma salvo em: {caminho_exec}")
    
    # Agrupamento Hierárquico
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels_hierarquico = agg.fit_predict(dist_matrix)
    score_hier = silhouette_score(dist_matrix, labels_hierarquico, metric='precomputed')
    print(f"    Silhueta do Hierárquico (k={n_clusters}): {score_hier:.4f}")
    return labels_hierarquico

# ==============================================================================
# 6. VISUALIZAÇÃO GRÁFICA DOS CLUSTERS EM 2D (PCA E MDS)
# ==============================================================================
def visualizar_clusters(df_original, df_features, dist_matrix, labels, medoid_indices, dir_saida):
    """
    Gera duas projeções bidimensionais:
    1. Projeção PCA Bidimensional (estilo idêntico ao artigo, com anotação dos medoides).
    2. Projeção MDS (baseada na matriz de distâncias de Gower).
    """
    print("\n[6] Gerando Projeções 2D dos Clusters (PCA e MDS)...")
    
    # --- 6.1 Projeção PCA (Exatamente como na Figura 3 do Artigo) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(X_scaled)
    
    cores = ['#6baed6', '#74c476', '#fb6a4a']
    nomes_clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    for i in range(len(nomes_clusters)):
        idx = (labels == i)
        plt.scatter(coords_pca[idx, 0], coords_pca[idx, 1], 
                    c=cores[i % len(cores)], label=nomes_clusters[i], alpha=0.75, s=45, edgecolors='none')
        
    # Medoides centrais
    medoids_coords = coords_pca[medoid_indices]
    plt.scatter(medoids_coords[:, 0], medoids_coords[:, 1],
                color='gold', edgecolor='black', s=240, marker='*', linewidth=1.2,
                label='Medoides Centrais', zorder=10)
    
    # Caixas de anotação com o ID do paciente de cada medoide
    offsets = [(-0.8, 0.4), (-0.7, 0.35), (-0.75, 0.35)]
    for idx_m, (x, y) in enumerate(medoids_coords):
        medoid_id = df_original.iloc[medoid_indices[idx_m]]['id_nodulo']
        plt.annotate(
            f"{medoid_id} (C{idx_m+1})",
            xy=(x, y),
            xytext=(x + offsets[idx_m][0], y + offsets[idx_m][1]),
            fontsize=9.5,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8)
        )
        
    plt.title('Projeção Bidimensional por PCA', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Componente Principal 1', fontsize=10)
    plt.ylabel('Componente Principal 2', fontsize=10)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    
    nome_pca = "projecao_pca_bidimensional.png"
    plt.savefig(os.path.join(dir_saida, nome_pca))
    plt.savefig(os.path.join(DIR_RESULTADOS, nome_pca))
    plt.savefig(os.path.join(dir_saida, "dispersao_clusters_pca_mds.png"))
    plt.savefig(os.path.join(DIR_RESULTADOS, "dispersao_clusters_pca_mds.png"))
    plt.close()
    print(f"    Projeção PCA 2D salva em: {os.path.join(dir_saida, nome_pca)}")
    
    # --- 6.2 Projeção MDS complementar ---
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
    coords_mds = mds.fit_transform(dist_matrix)
    
    plt.figure(figsize=(8, 5.5), dpi=300)
    for i in range(len(nomes_clusters)):
        idx = (labels == i)
        plt.scatter(coords_mds[idx, 0], coords_mds[idx, 1], 
                    c=cores[i % len(cores)], label=nomes_clusters[i], alpha=0.75, s=50, edgecolors='white', linewidth=0.5)
        
    medoids_mds = coords_mds[medoid_indices]
    plt.scatter(medoids_mds[:, 0], medoids_mds[:, 1],
                color='gold', edgecolor='black', s=220, marker='*', linewidth=1.5,
                label='Medoides (Casos Representativos)', zorder=10)
    
    plt.title('Projeção Espacial 2D dos Nódulos (MDS com Distância de Gower)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Dimensão 1 (MDS)', fontsize=10)
    plt.ylabel('Dimensão 2 (MDS)', fontsize=10)
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    
    nome_mds = "projecao_mds_gower.png"
    plt.savefig(os.path.join(dir_saida, nome_mds))
    plt.savefig(os.path.join(DIR_RESULTADOS, nome_mds))
    plt.close()
    print(f"    Projeção MDS 2D salva em: {os.path.join(dir_saida, nome_mds)}")

# ==============================================================================
# 7. CARACTERIZAÇÃO CLÍNICA DOS CLUSTERS E VALIDAÇÃO EXTERNA
# ==============================================================================
def caracterizar_clusters(df_original, labels, medoid_indices, dir_saida):
    """
    Gera tabela descritiva clínica cruzando cada cluster com as variáveis morfológicas,
    apresenta os dados dos medoides reais e calcula a taxa real de malignidade (desfecho).
    """
    print("\n[7] Caracterizando o Perfil Clínico e Validação Externa...")
    df_analise = df_original.copy()
    df_analise['cluster'] = labels
    
    n_clusters = len(np.unique(labels))
    resumo = []
    
    for c in range(n_clusters):
        sub = df_analise[df_analise['cluster'] == c]
        medoid_pac = df_original.iloc[medoid_indices[c]]
        
        info = {
            'Cluster': f"Cluster {c+1}",
            'N_Nodulos': len(sub),
            'Proporcao (%)': round(len(sub) / len(df_analise) * 100, 1),
            'Medoid_ID': medoid_pac['id_nodulo'],
            'Idade_Media (anos)': round(sub['idade'].mean(), 1),
            'Tamanho_Medio (cm)': round(sub['usg_tamanho_nodulo_maior'].mean(), 2),
            'Pct_Bethesda_IV (%)': round((sub['citologia'] == 4).mean() * 100, 1),
            'Pct_Solido (%)': round((sub['usg_conteudo_nodulo'] == 2).mean() * 100, 1),
            'Pct_Hipoecoico (%)': round((sub['usg_ecogenicidade_nodulo'].isin([2, 3])).mean() * 100, 1),
            'Pct_Margem_Irregular (%)': round((sub['usg_contorno_nodulo'].isin([2, 3])).mean() * 100, 1),
            'Pct_Mais_Alto_que_Largo (%)': round((sub['usg_formato'] == 2).mean() * 100, 1),
            'Pct_Microcalcificacoes (%)': round((sub['usg_microcalcificacoes'] == 1).mean() * 100, 1),
            'Pct_ACR_TR5_Alto_Risco (%)': round((sub['usg_acr_nodulo'] == 5).mean() * 100, 1),
            
            # --- VALIDAÇÃO EXTERNA (VARIÁVEL DESFECHO HISTOPATOLÓGICO) ---
            'Taxa_Cancer_Malignidade (%)': round((sub['cancer_de_tireoide'] == 1).mean() * 100, 1),
        }
        resumo.append(info)
        
    df_resumo = pd.DataFrame(resumo)
    
    # Salva CSV de resumo na pasta da execução e na raiz de resultados
    nome_csv_resumo = "perfil_clinico_clusters.csv"
    caminho_csv_exec = os.path.join(dir_saida, nome_csv_resumo)
    caminho_csv_raiz = os.path.join(DIR_RESULTADOS, nome_csv_resumo)
    df_resumo.to_csv(caminho_csv_exec, sep=";", index=False, encoding="utf-8")
    df_resumo.to_csv(caminho_csv_raiz, sep=";", index=False, encoding="utf-8")
    print(f"    Tabela descritiva exportada em: {caminho_csv_exec}")
    
    # Impressão legível no terminal
    print("\n" + "="*85)
    print("RESUMO EXECUTIVO DO PERFIL CLÍNICO DOS CLUSTERS (VALIDAÇÃO CLÍNICA)")
    print("="*85)
    colunas_exibir = [
        'Cluster', 'N_Nodulos', 'Proporcao (%)', 'Tamanho_Medio (cm)', 
        'Pct_Bethesda_IV (%)', 'Pct_Hipoecoico (%)', 'Pct_Microcalcificacoes (%)', 
        'Taxa_Cancer_Malignidade (%)'
    ]
    print(df_resumo[colunas_exibir].to_string(index=False))
    print("="*85)
    
    # Salva base original completa com a coluna cluster atribuída
    nome_base_clusters = "nodulos_com_clusters_atribuidos.csv"
    caminho_base_exec = os.path.join(dir_saida, nome_base_clusters)
    caminho_base_raiz = os.path.join(DIR_RESULTADOS, nome_base_clusters)
    df_analise.to_csv(caminho_base_exec, sep=";", index=False, encoding="utf-8")
    df_analise.to_csv(caminho_base_raiz, sep=";", index=False, encoding="utf-8")
    print(f"    Base com clusters gravada em: {caminho_base_exec}")

# ==============================================================================
# FLUXO PRINCIPAL
# ==============================================================================
def main():
    import sys
    if len(sys.argv) > 1:
        caminho_dados = sys.argv[1]
    elif os.path.exists("nodulos_sinteticos_500.csv"):
        caminho_dados = "nodulos_sinteticos_500.csv"
    else:
        caminho_dados = "nodulos_sinteticos_200.csv"
    
    # Cria pasta com a data e hora desta execução específica
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_execucao = os.path.join(DIR_RESULTADOS, f"execucao_{timestamp}")
    os.makedirs(dir_execucao, exist_ok=True)
    print(f"\n[INFO] Nova pasta criada para esta execução: '{dir_execucao}'")
    
    # 1. Carregamento e Preparação
    df_original, df_features, is_categorical, colunas_features, colunas_desfecho = carregar_e_preparar_dados(caminho_dados)
    
    # 2. Matriz de Gower
    dist_matrix = calcular_matriz_gower(df_features, is_categorical)
    
    # 3. Avaliação da Silhueta para escolha de k
    melhor_k = avaliar_numero_clusters(dist_matrix, dir_execucao, k_min=2, k_max=6)
    
    # Fixaremos k=3 (ou o melhor k indicado) para explorar a estratificação clínica de risco
    k_escolhido = 3 if melhor_k in [2, 3, 4] else melhor_k
    
    # 4. Ajuste do K-Medoids
    labels_kmedoids, medoid_indices, score = executar_kmedoids(dist_matrix, n_clusters=k_escolhido)
    
    # 5. Agrupamento Hierárquico
    executar_hierarquico(dist_matrix, dir_execucao, n_clusters=k_escolhido)
    
    # 6. Visualização 2D (PCA e MDS)
    visualizar_clusters(df_original, df_features, dist_matrix, labels_kmedoids, medoid_indices, dir_execucao)
    
    # 7. Perfil Clínico e Validação Externa
    caracterizar_clusters(df_original, labels_kmedoids, medoid_indices, dir_execucao)
    
    print("\n[OK] Pipeline concluído com absoluto sucesso!")
    print(f"     -> Histórico gravado na pasta: '{dir_execucao}'")
    print(f"     -> Arquivos mais recentes atualizados em: '{DIR_RESULTADOS}/'")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
