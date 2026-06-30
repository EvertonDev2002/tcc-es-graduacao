# Rastreio Emocional Automatizado e Extração de RPD

**Trabalho de Conclusão de Curso — Engenharia de Software**

Este repositório contém o código-fonte de um sistema de Processamento de Linguagem Natural (PLN) voltado para a saúde mental. O sistema automatiza o Registro de Pensamentos Disfuncionais (RPD) a partir de diários textuais (sintético) de pacientes com Transtorno do Espectro Autista (TEA), mitigando as barreiras comunicacionais da alexitimia.

---

## 1. Arquitetura do Sistema

O projeto implementa uma **Arquitetura de Pipeline Híbrido**, processando textos longos através de duas frentes simultâneas:

* **Paradigma Neural (Deep Learning):** Utiliza o modelo Transformer `RoBERTa` (caramelo-smile-2) adaptado com um algoritmo de *Sliding Window* (Janelamento Deslizante) para contornar o limite arquitetural de 512 *tokens*. Responsável pela classificação da polaridade emocional global da narrativa.
* **Paradigma Simbólico (Extração Baseada em Regras):** Utiliza a biblioteca `spaCy` para o processamento morfossintático. O motor rastreia Expressões Multipalavras (MWEs) para traduzir queixas somáticas em dados emocionais (solucionando o viés de predição de "Falsos Neutros"), e aplica Reconhecimento de Entidades Nomeadas (NER) para mapear gatilhos ambientais.
* **Interface Clínica:** Um *dashboard* interativo construído em `streamlit`, concebido para execução local leve e preservação rigorosa do sigilo dos dados do paciente.

---

## 2. Estrutura de Diretórios

A organização do código segue o padrão de engenharia de dados:

```text
.
├── app.py                      # Ponto de entrada do Dashboard Streamlit
├── data/                       # Armazenamento de dados
│   ├── raw/                    # Corpus sintético gerado (não modificado)
│   └── processed/              # Corpus enriquecido após inferência do pipeline
├── reports/                    # Gráficos analíticos e diagramas gerados (Matriz, Sankey)
├── scripts/                    # Scripts de automação e avaliação
│   ├── corpus_generation.py    # Gerador de diários sintéticos com perfis clínicos
│   ├── run_pipeline.py         # Orquestrador do processamento de inferência PLN
│   └── evaluate_metrics.py     # Avaliador estatístico do modelo (Scikit-Learn)
├── src/                        # Código-fonte principal
│   ├── configs/                # Arquivos de configuração, constantes e léxicos
│   └── rpd_extractor/          # Motor de NLP simbólico e extração de regras
├── pyproject.toml              # Dependências e metadados do projeto
└── uv.lock                     # Ficheiro de bloqueio do gestor de pacotes
```

## 3. Pré-requisitos e Instalação

O projeto requer `Python` >= 3.12 e utiliza o gerenciador de pacotes `uv`.

Passo 1: Clonar repositório
```bash

git clone git@github.com:EvertonDev2002/tcc-es-graduacao.git

```

Passo 2: Sincronize o ambiente virtual e instale as dependências.
```bash

uv sync

```

Passo 3: Instale o projeto localmente em modo editável
```bash

uv pip install -e .

```

Passo 4: Baixar o modelo de linguagem em português do spacy.
```bash

# Utilize
uv run python -m spacy download pt_core_news_lg

# Caso ocorra algum error, tente baixar diretamente
uv pip install  https://github.com/explosion/spacy-models/releases/download/pt_core_news_lg-3.8.0/pt_core_news_lg-3.8.0-py3-none-any.whl

```

## 4. Guia de Execução

Passo 1: Geração de Dados
```bash

uv run python scripts/corpus_generation.py

```

Passo 2: Execução do Pipeline Híbrido
```bash

uv run python scripts/run_pipeline.py

```

Passo 3: Avaliação de Métricas
```bash

uv run python scripts/evaluate_metrics.py

```

Passo 4: Dashboard
```bash

uv run streamlit run app.py

```

## 5. Autoria e Licença

Autor: Éverton

Este projeto tem fins estritamente académicos de validação de conceito e foi desenvolvido como requisito para a obtenção do grau de Engenheiro de Software.
