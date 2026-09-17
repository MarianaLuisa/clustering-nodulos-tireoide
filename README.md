# Análise do Comportamento de Nódulos de Tireoide com Citologia Indeterminada por Meio de Algoritmos de Agrupamento de Dados

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-orange.svg)](https://scikit-learn.org/)
[![Clustering](https://img.shields.io/badge/Clustering-K--Medoids%20%7C%20Hierarchical-green.svg)]()
[![Metric](https://img.shields.io/badge/Distance-Gower%20Metric-purple.svg)]()
[![Institution](https://img.shields.io/badge/UFCSPA-Porto%20Alegre%20--%20RS-red.svg)](https://www.ufcspa.edu.br/)

Este repositório contém a implementação do pipeline computacional de **Aprendizado de Máquina Não Supervisionado (*Clustering*)** desenvolvido para a estratificação e análise fenotípica de **nódulos tireoidianos com citologia indeterminada (Bethesda III e IV)**, trabalho integrante do Trabalho de Conclusão de Curso (TCC) em Informática Biomédica na **Universidade Federal de Ciências da Saúde de Porto Alegre (UFCSPA)**.

O projeto implementa uma abordagem estatisticamente rigorosa para **dados clínicos mistos** (variáveis contínuas, ordinais e nominais), utilizando a **Matriz de Dissimilaridade de Gower**, particionamento interpretável por **K-Medoids (algoritmo PAM)**, validação por **Agrupamento Hierárquico Aglomerativo**, projeção não supervisionada em 2D (**PCA e MDS**) e validação clínica externa cega contra desfechos histopatológicos cirúrgicos.

---

## Sumário

- [1. Contexto e Motivação Clínica](#1-contexto-e-motivação-clínica)
- [2. Fundamentação Teórica e Desafios Metodológicos](#2-fundamentação-teórica-e-desafios-metodológicos)
- [3. Técnicas Aplicadas e Modelagem Matemática](#3-técnicas-aplicadas-e-modelagem-matemática)
  - [3.1. Matriz de Dissimilaridade de Gower](#31-matriz-de-dissimilaridade-de-gower)
  - [3.2. K-Medoids com Algoritmo PAM](#32-k-medoids-com-algoritmo-pam)
  - [3.3. Agrupamento Hierárquico Aglomerativo](#33-agrupamento-hierárquico-aglomerativo)
  - [3.4. Validação Interna: Coeficiente de Silhueta](#34-validação-interna-coeficiente-de-silhueta)
  - [3.5. Redução Dimensional e Visualização Espacial (PCA & MDS)](#35-redução-dimensional-e-visualização-espacial-pca--mds)
  - [3.6. Validação Clínica Externa (Prevenção de Data Leakage)](#36-validação-clínica-externa-prevenção-de-data-leakage)
- [4. Resultados e Perfis Clínicos Descobertos](#4-resultados-e-perfis-clínicos-descobertos)
- [5. Dicionário de Variáveis de Entrada](#5-dicionário-de-variáveis-de-entrada)
- [6. Estrutura do Repositório](#6-estrutura-do-repositório)
- [7. Guia de Instalação e Execução](#7-guia-de-instalação-e-execução)
- [8. Referências Bibliográficas](#8-referências-bibliográficas)
- [9. Autoria](#9-autoria)

---

## 1. Contexto e Motivação Clínica

A Punção Aspirativa por Agulha Fina orientada por Ultrassonografia (PAAF-US) é o exame de escolha para a triagem diagnóstica da doença nodular tireoidiana. Contudo, em aproximadamente 20% a 25% das punções, a análise citológica resulta em categorias indeterminadas do Sistema Bethesda:
* **Bethesda III**: Atipia de Significado Indeterminado / Lesão Folicular de Significado Indeterminado (*AUS/FLUS*).
* **Bethesda IV**: Neoplasia Folicular ou Suspeita de Neoplasia Folicular (*FN/SFN*).

Nestes casos, o risco estimado de malignidade varia entre 10% e 30%. Devido à impossibilidade de avaliar a invasão capsular ou vascular na citologia convencional, a conduta rotineira frequentemente encaminha o paciente para **tireoidectomia diagnóstica**. O resultado histopatológico pós-operatório revela que **entre 70% e 80% dos procedimentos cirúrgicos são realizados em lesões de caráter inteiramente benigno**, submetendo pacientes a riscos operatórios, reposição hormonal vitalícia e elevados custos hospitalares.

O uso do **Aprendizado de Máquina Não Supervisionado** permite analisar a morfologia combinada ultrassonográfica e clínica, descobrindo subgrupos naturais e estratificando o risco sem depender de rótulos prévios.

```
                  Nódulo com Citologia Indeterminada
                         (Bethesda III / IV)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   Conduta Tradicional                             Abordagem Proposta
   (Cirurgia Diagnóstica)                      (Pipeline Não Supervisionado)
         │                                                 │
   70% a 80% ressecções benignas                Descoberta de Subperfis Naturais
   (Tratamento invasivo desnecessário)         (Baixo Risco vs Alto Risco vs Folicular)
                                                           │
                                                Suporte à Decisão Clínica
                                                (Vigilância, Biologia ou Cirurgia)
```

---

## 2. Fundamentação Teórica e Desafios Metodológicos

Em registros médicos de nódulos tireoidianos, os atributos coletados são inerentemente **heterogêneos (dados mistos)**:
- **Contínuos**: Idade (anos), Maior diâmetro do nódulo (cm).
- **Binários / Nominais**: Sexo, Microcalcificações, Halo, Linfonodomegalias.
- **Politômicos / Ordinais**: Ecogenicidade (isoecoico a muito hipoecoico), Margens (regular a espiculada), Sistema ACR TI-RADS (TR2 a TR5).

### Por que o K-Means e a Distância Euclidiana falham em dados mistos?
O algoritmo clássico K-Means e métricas euclidianas assumem espaços vetoriais contínuos lineares:
1. **Distorção geométrica em variáveis discretas**: Tratar variáveis categóricas codificadas numericamente (ex.: $1, 2, 3$) como se possuíssem métrica linear introduz uma ordenação e espaçamento artificiais que corrompem as distâncias relativas.
2. **Centróides irreais ("Pacientes Médios Virtuais")**: O K-Means calcula a média matemática dos pontos de cada grupo. Em dados mistos, a média resulta em valores biologicamente inexistentes (ex.: "sexo = 1,43", "presença de microcalcificação = 0,38"), destruindo a interpretabilidade clínica.
3. **Sensibilidade extrema a outliers**: A soma dos erros quadráticos distancia os centróides na presença de lesões atípicas.

Para solucionar essas limitações, este projeto acopla a **Métrica de Gower** ao algoritmo **K-Medoids (PAM)**.

---

## 3. Técnicas Aplicadas e Modelagem Matemática

```
                   ┌──────────────────────────────────────┐
                   │  Variáveis Clínicas e Ultrassonográficas │
                   │  (13 Atributos: Contínuos e Categóricos) │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │   Matriz de Dissimilaridade de Gower  │
                   │   D ∈ [0, 1] com Simetria e Reflexividade│
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │ Validação Interna (Silhueta p/ k=2..6)│
                   │ Determinação do k Ótimo Global (k=3) │
                   └──────────────────┬───────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     ┌───────────────────────────┐         ┌───────────────────────────┐
     │    K-Medoids (Método PAM) │         │   Hierárquico Aglomerativo│
     │ Medoides Reais de Pacientes│         │       Average Linkage     │
     └─────────────┬─────────────┘         └─────────────┬─────────────┘
                   │                                     │
                   ▼                                     ▼
     ┌───────────────────────────┐         ┌───────────────────────────┐
     │  Projeções Espaciais 2D   │         │  Dendrograma de Coerência │
     │       (PCA e MDS)         │         │        Topológica         │
     └─────────────┬─────────────┘         └───────────────────────────┘
                   │
                   ▼
     ┌───────────────────────────────────────────────────┐
     │   Validação Histopatológica Cega e Perfilamento   │
     │    (Estratificação de Risco sem Data Leakage)     │
     └───────────────────────────────────────────────────┘
```

### 3.1. Matriz de Dissimilaridade de Gower
A métrica proposta por **Gower (1971)** calcula a dissimilaridade $D_{ij}$ entre dois nódulos $i$ e $j$ descritos por $p$ atributos mistos:

$$S_{ij} = \frac{\sum_{k=1}^{p} w_k \cdot s_{ijk}}{\sum_{k=1}^{p} w_k}, \quad D_{ij} = 1 - S_{ij}$$

onde $w_k = 1$ é o peso uniforme conferido a cada característica, e a similaridade parcial $s_{ijk}$ é calculada de acordo com o tipo da variável:

* **Para atributos contínuos** (idade e tamanho):
  $$s_{ijk} = 1 - \frac{|x_{ik} - x_{jk}|}{R_k}$$
  onde $R_k = \max(x_k) - \min(x_k)$ é a amplitude total observada da variável $k$.

* **Para atributos categóricos, nominais e binários**:
  $$s_{ijk} = \begin{cases} 1, & \text{se } x_{ik} = x_{jk} \\ 0, & \text{se } x_{ik} \neq x_{jk} \end{cases}$$

**Garantias Numéricas Implementadas:**
No pipeline, a matriz resultante passa por pós-processamento de simetrização e ajuste diagonal estrito:
- Reflexividade: $D_{ii} = 0, \quad \forall i$.
- Simetria exata: $D_{ij} = D_{ji} = \frac{D_{ij} + D_{ji}}{2}$.
- Delimitação estrita no intervalo unitário: $0 \le D_{ij} \le 1$.

---

### 3.2. K-Medoids com Algoritmo PAM
Diferente do K-Means, o **K-Medoids** escolhe obrigatoriamente **instâncias reais da base de dados** como representantes centrais de cada grupo (denominados *medoids*). 

O pipeline utiliza a variante **PAM (*Partitioning Around Medoids*)** de Kaufman e Rousseeuw (1990):
1. **Fase BUILD**: Seleciona iterativamente $k$ instâncias cuja soma das dissimilaridades com todos os outros pontos é minimizada.
2. **Fase SWAP**: Avalia pares de medoides atuais $m$ e não-medoides $o$. A troca de centro só é consolidada se a redução no custo global de dissimilaridade for estritamente favorável:
   $$\min \sum_{i=1}^{n} D(x_i, m_{\text{mais\_próximo}})$$

**Vantagem Clínica:** Cada cluster tem como referência um **paciente real**, permitindo aos médicos inspecionar integralmente o histórico clínico e o laudo ultrassonográfico do caso-índice representativo.

---

### 3.3. Agrupamento Hierárquico Aglomerativo
Para verificar a estabilidade das fronteiras de agrupamento e a integridade da partição, o pipeline aplica o **Agrupamento Hierárquico Aglomerativo** com critério de **Ligação Média (*Average Linkage*)**:

$$d(u, v) = \sum_{i \in u} \sum_{j \in v} \frac{D_{ij}}{|u| \cdot |v|}$$

Essa abordagem computa a média de dissimilaridade entre todos os pares de observações de dois clusters $u$ e $v$, gerando uma árvore taxonômica (dendrograma) sem forçar formas esféricas a priori.

---

### 3.4. Validação Interna: Coeficiente de Silhueta
A determinação objetiva da quantidade ótima de clusters ($k$) é conduzida pelo **Coeficiente de Silhueta Médio**:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

onde:
* $a(i)$ é a distância média do nódulo $i$ aos demais membros do seu próprio cluster (coesão intra-cluster).
* $b(i)$ é a distância média do nódulo $i$ aos membros do cluster vizinho mais próximo (separação inter-cluster).

O valor varia entre $-1$ e $+1$. O pipeline testa sistematicamente o intervalo $k \in [2, 6]$. O pico do escore define o melhor número de partições.

---

### 3.5. Redução Dimensional e Visualização Espacial (PCA & MDS)
Para viabilizar a interpretação geométrica dos agrupamentos de 13 dimensões no plano cartesiano $\mathbb{R}^2$:
1. **Projeção PCA Bidimensional**: Reduz as componentes principais mantendo a máxima variância, exibindo as observações com marcação e destaque individual das anotações dos medoides centrais.
2. **Multidimensional Scaling (MDS)**: Realiza uma projeção não linear diretamente sobre a matriz de Gower, minimizando a função de *Stress*:
   $$\text{Stress} = \sqrt{\sum_{i < j} \left( D_{ij} - \|\mathbf{z}_i - \mathbf{z}_j\| \right)^2}$$
   onde $\mathbf{z}_i, \mathbf{z}_j \in \mathbb{R}^2$ são as coordenadas no plano reduzido.

---

### 3.6. Validação Clínica Externa (Prevenção de Data Leakage)
Um rigor metodológico central deste projeto é a **completa separação entre o aprendizado do modelo e a validação do desfecho**:
- As variáveis histopatológicas pós-cirúrgicas (`cancer_de_tireoide` e `diagnostico_definitivo`) foram **estritamente excluídas** do cômputo da matriz de Gower e de todo o processo de agrupamento.
- Apenas após a atribuição cega dos rótulos de cluster ($C_1, C_2, C_3$) pelo K-Medoids, calculou-se a proporção observada de lesões malignas comprovadas em cada agrupamento, provando matematicamente que o algoritmo capturou padrões fenotípicos de risco real.

---

## 4. Resultados e Perfis Clínicos Descobertos

### 4.1. Validação Interna do Número Ótimo de Grupos
A análise do Coeficiente de Silhueta Médio em função de $k$ confirmou $k=3$ como a partição ótima global:

| Número de Grupos ($k$) | Coeficiente de Silhueta Médio |
| :---: | :---: |
| $k = 2$ | 0,2494 |
| **$k = 3$** | **0,2545 (Pico Ótimo)** |
| $k = 4$ | 0,1842 |
| $k = 5$ | 0,1774 |
| $k = 6$ | 0,1660 |

![Curva de Silhueta](resultados/curva_silhueta_kmedoids.png)

O agrupamento hierárquico alcançou silhueta média de **0,3332**, e a inspeção do dendrograma corroborou a divisão natural da coorte em 3 ramos principais:

![Dendrograma Hierárquico](resultados/dendrograma_hierarquico.png)

---

### 4.2. Caracterização Clínica dos Clusters e Validação com Malignidade

| Métrica / Parâmetro Clínico | Cluster 1 (*Baixo Risco*) | Cluster 2 (*Alto Risco USG*) | Cluster 3 (*Padrão Folicular*) |
| :--- | :---: | :---: | :---: |
| **Volume de Nódulos / Proporção** | 227 (45,4%) | 109 (21,8%) | 164 (32,8%) |
| **Paciente Medoide Central** | `NOD_043` | `NOD_054` | `NOD_088` |
| **Idade Média (anos)** | 51,2 | 53,0 | 53,6 |
| **Maior Diâmetro Médio (cm)** | 2,37 | 2,06 | 2,94 |
| **Predomínio Citológico** | 67,0% Bethesda III | 57,8% Bethesda III | **79,3% Bethesda IV** |
| **Consistência Sólida (%)** | 44,9% (Mistos/Císticos) | **100,0%** | 82,3% |
| **Ecotextura Hipoecoica (%)** | 21,1% | **100,0%** | 68,3% |
| **Margens Irregulares/Espiculadas (%)** | 3,1% | **100,0%** | 29,9% |
| **Formato Mais Alto que Largo (%)** | 0,0% | **66,1%** | 3,0% |
| **Microcalcificações Presentes (%)** | 2,2% | **82,6%** | 12,2% |
| **Classificação ACR TI-RADS 5 (%)** | 0,0% | **82,6%** | 8,5% |
| **TAXA REAL DE CÂNCER (Validação Externa)** | **9,7% (Baixo Risco)** | **68,8% (Alto Risco)** | **28,0% (Risco Intermediário)** |

---

### 4.3. Interpretação Médica dos Perfis

* **Cluster 1 — Perfil de Baixo Risco (45,4% da coorte)**:
  - Nódulos predominantemente mistos ou isoecoicos, margens bem delineadas, sem microcalcificações e com escore ACR TI-RADS baixo (TR2-TR3).
  - Taxa de malignidade comprovada de apenas **9,7%**.
  - **Implicação Prática**: Fortes candidatos a **acompanhamento ultrassonográfico conservador (vigilância ativa)**, com potencial de evitar aproximadamente 45% das ressecções cirúrgicas diagnósticas desnecessárias.

* **Cluster 2 — Perfil de Alto Risco / Suspeita Papilífera (21,8% da coorte)**:
  - Nódulos 100% sólidos e hipoecoicos, acentuada frequência de microcalcificações (82,6%), margens espiculadas/irregulares (100%), crescimento vertical (mais alto que largo: 66,1%) e ACR TI-RADS 5 (82,6%).
  - Taxa de malignidade de **68,8%** (carcinoma papilífero clássico).
  - **Implicação Prática**: Nódulos com indicação mandatória de **intervenção cirúrgica ágil**, mesmo com laudo citológico indeterminado.

* **Cluster 3 — Perfil de Padrão Folicular (32,8% da coorte)**:
  - Nódulos de dimensões aumentadas (média de 2,94 cm), predomínio maciço de citologia **Bethesda IV (79,3%)**, presença frequente de halo periférico e margens regulares.
  - Taxa de malignidade de **28,0%**, condizente com a literatura para neoplasias foliculares (adenomas vs. carcinomas foliculares / NIFTP).
  - **Implicação Prática**: Grupo ideal para **investigação molecular dirigida (painéis genéticos de expressão)** ou avaliação complementar de invasão tecidual.

---

### 4.4. Projeções Espaciais dos Agrupamentos

#### Projeção PCA Bidimensional com Medoides Reais
Os pacientes protótipos reais identificados pelo K-Medoids (`NOD_043`, `NOD_054` e `NOD_088`) encontram-se representados pelas estrelas douradas:

![Projeção PCA](resultados/projecao_pca_bidimensional.png)

#### Projeção Não Linear por Multidimensional Scaling (MDS)
Preserva a topologia direta da Matriz de Dissimilaridade de Gower:

![Projeção MDS](resultados/projecao_mds_gower.png)

---

## 5. Dicionário de Variáveis de Entrada

O pipeline consome um arquivo estruturado delimitado por ponto e vírgula (`;`), contendo 13 variáveis de agrupamento e os desfechos para validação:

| Campo | Tipo | Descrição Clínica e Codificação |
| :--- | :---: | :--- |
| `id_paciente` | Texto | Identificador anonimizado do paciente (ex.: `PAC_001`) |
| `id_nodulo` | Texto | Identificador anonimizado do nódulo (ex.: `NOD_001`) |
| `idade` | Numérico | Idade do paciente no momento da punção (anos) |
| `sexo` | Categórico | Sexo biológico (`1`: Feminino, `2`: Masculino) |
| `citologia` | Categórico | Sistema Bethesda (`3`: Bethesda III, `4`: Bethesda IV) |
| `usg_tamanho_nodulo_maior` | Numérico | Maior eixo do nódulo mensurado ao ultrassom (em cm) |
| `usg_conteudo_nodulo` | Categórico | Composição (`1`: Cístico/Espongiforme, `2`: Sólido, `3`: Misto) |
| `usg_ecogenicidade_nodulo` | Categórico | Ecogenicidade (`1`: Iso/Hiperecoico, `2`: Hipoecoico, `3`: Muito Hipoecoico) |
| `usg_contorno_nodulo` | Categórico | Margens (`1`: Regular, `2`: Irregular, `3`: Espiculada) |
| `usg_formato` | Categórico | Geometria (`1`: Mais largo que alto, `2`: Mais alto que largo) |
| `usg_halo_nodulo` | Categórico | Halo periférico (`1`: Presente/Completo, `2`: Ausente/Incompleto, `99`: Não avaliado) |
| `usg_fluxo_nodulo` | Categórico | Padrão Doppler (`1`: Periférico, `2`: Misto, `3`: Central) |
| `usg_microcalcificacoes` | Binário | Focos ecogênicos pontuais (`0`: Ausente, `1`: Presente) |
| `usg_linfonodos` | Binário | Linfonodos cervicais regionais (`0`: Habitual/Normal, `1`: Suspeito) |
| `usg_acr_nodulo` | Categórico | Classificação ACR TI-RADS estimada (`2`: TR2 a `5`: TR5) |
| `cancer_de_tireoide` | Binário (Externo) | Padrão-ouro histopatológico (`0`: Benigno, `1`: Maligno) |
| `diagnostico_definitivo` | Texto (Externo) | Diagnóstico anatomopatológico conclusivo pós-ressecção |

---

## 6. Estrutura do Repositório

```text
├── .gitignore                      # Regras de exclusão (dados sensíveis, arquivos temporários)
├── README.md                       # Documentação técnica, fundamentação teórica e resultados
├── requirements.txt                # Especificação de dependências do ambiente Python
├── pipeline_clustering.py          # Implementação completa do pipeline de agrupamento e validação
└── resultados/                     # Resultados gráficos e métricas consolidadas
    ├── curva_silhueta_kmedoids.png
    ├── dendrograma_hierarquico.png
    ├── projecao_pca_bidimensional.png
    ├── projecao_mds_gower.png
    ├── dispersao_clusters_pca_mds.png
    └── perfil_clinico_clusters.csv # Tabela agregada com as métricas e taxas de câncer por cluster
```

---

## 7. Guia de Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior
- Git

### 1. Clonar o repositório
```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
cd NOME_DO_REPOSITORIO
```

### 2. Criar e ativar um ambiente virtual
```bash
# No Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# No Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Executar o pipeline de agrupamento
Para executar o pipeline com seu arquivo de dados:
```bash
python pipeline_clustering.py caminho/para/sua_base_de_dados.csv
```

O script executará automaticamente:
1. Validação dos tipos de variáveis e preparação da matriz de features;
2. Cálculo da matriz de Gower;
3. Determinação do $k$ ótimo via Coeficiente de Silhueta;
4. Agrupamento por K-Medoids (PAM) e Hierárquico;
5. Renderização das projeções espaciais 2D com destaque para os medoides reais;
6. Validação clínica externa contra o desfecho histopatológico;
7. Exportação de todos os gráficos e do sumário descritivo em `resultados/perfil_clinico_clusters.csv`.

---

## 8. Referências Bibliográficas

* **ALEXANDER, E. K. et al.** Preoperative Diagnosis of Benign Thyroid Nodules with Indeterminate Cytology. *The New England Journal of Medicine*, v. 367, n. 8, p. 705–715, 2012. DOI: [10.1056/NEJMoa1203208](https://doi.org/10.1056/NEJMoa1203208).
* **DURANTE, C. et al.** The Diagnosis and Management of Thyroid Nodules: A Review. *JAMA*, v. 319, n. 9, p. 914–924, 2018. DOI: [10.1001/jama.2018.0898](https://doi.org/10.1001/jama.2018.0898).
* **GOWER, J. C.** A General Coefficient of Similarity and Some of Its Properties. *Biometrics*, v. 27, n. 4, p. 857–871, 1971.
* **KAUFMAN, L.; ROUSSEEUW, P. J.** *Finding Groups in Data: An Introduction to Cluster Analysis*. New York: John Wiley & Sons, 1990.
* **KIM, N. E. et al.** Bethesda III and IV Thyroid Nodules Managed Nonoperatively After Molecular Testing With Afirma GSC or Thyroseq v3. *Journal of Clinical Endocrinology and Metabolism*, 2023. DOI: [10.1210/clinem/dgad191](https://doi.org/10.1210/clinem/dgad191).
* **NIENOW, D. et al.** Aplicabilidade clínica da avaliação molecular nos nódulos de tireoide com citologia indeterminada em uma amostra no Sul do Brasil. Projeto de pesquisa clínica, UFCSPA, 2024.
* **PREUD’HOMME, G. et al.** Head-to-Head Comparison of Clustering Methods for Heterogeneous Data: A Simulation-Driven Benchmark. *Scientific Reports*, v. 11, p. 4202, 2021. DOI: [10.1038/s41598-021-83340-8](https://doi.org/10.1038/s41598-021-83340-8).
* **ROUSSEEUW, P. J.** Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, v. 20, p. 53–65, 1987.
* **WERNER, E. et al.** Explainable Hierarchical Clustering for Patient Subtyping and Risk Prediction. *Experimental Biology and Medicine*, v. 248, p. 2547–2559, 2023.

---

## 9. Autoria

**Mariana Silva**  
Universidade Federal de Ciências da Saúde de Porto Alegre (UFCSPA)  
Porto Alegre – RS – Brasil  
E-mail: `mariana.silva@ufcspa.edu.br`
