"""Avaliação estatística do classificador RoBERTa no corpus RPD e geração de gráficos."""

import logging
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.configs import PathConfig

logger = logging.getLogger(__name__)

class AvaliadorDeModelo:
    """Calcula métricas e gera gráficos avançados de desempenho do modelo RoBERTa."""

    def __init__(self):
        # Gabarito de polaridade baseado na trajetória esperada das Personas
        self.gabarito_personas = {
            "Persona_A": "POSITIVE",
            "Persona_B": "NEGATIVE",
            "Persona_C": "NEGATIVE",
            "Persona_D": "NEUTRAL",
            "Persona_E": "NEGATIVE",
        }

        # Garante que a pasta reports existe
        self.reports_dir = PathConfig.BASE_DIR / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def carregar_resultados(self) -> pd.DataFrame:
        caminho = PathConfig.CORPUS_PROCESSED_PATH
        if not caminho.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {caminho}.")
        return pd.read_csv(caminho)

    def _preparar_dados(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        df["Polaridade_Esperada"] = df["persona"].map(self.gabarito_personas)

        # Suporte para a transição de nomenclatura de BERT para RoBERTa no CSV
        coluna_predicao = "Polaridade (RoBERTa)"
        if coluna_predicao not in df.columns and "Polaridade (BERT)" in df.columns:
            coluna_predicao = "Polaridade (BERT)"

        df_valido = df.dropna(subset=["Polaridade_Esperada", coluna_predicao])
        return df_valido["Polaridade_Esperada"].tolist(), df_valido[coluna_predicao].tolist()

    # ==========================================
    # RELATÓRIO DE TERMINAL
    # ==========================================

    def exibir_relatorio(self, y_true: list[str], y_pred: list[str]) -> None:
        """Calcula e imprime acurácia, precision, recall e f1-score."""
        acuracia = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        separador = "=" * 52
        print(f"\n{separador}")
        print("  RELATÓRIO DE AVALIAÇÃO DE DESEMPENHO — RoBERTa")
        print(separador)
        print(f"  Acurácia Global  : {acuracia:.4f}")
        print(f"  Precision (Macro) : {precision:.4f}")
        print(f"  Recall (Macro)   : {recall:.4f}")
        print(f"  F1-Score (Macro) : {f1:.4f}")
        print(f"  {'-' * 48}")
        print("  Detalhamento por classe:")
        print()
        print(classification_report(y_true, y_pred, zero_division=0))
        print(separador)

        logger.info(
            "Métricas — Acurácia: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            acuracia,
            f1,
            precision,
            recall,
        )

    # ==========================================
    # GERAÇÃO DE GRÁFICOS
    # ==========================================

    def gerar_matriz_normalizada(self, y_true: list[str], y_pred: list[str]) -> None:
        """Gera a Matriz de Confusão com números absolutos e percentagens."""
        classes = sorted(list(set(y_true + y_pred)))

        # Calcula matriz absoluta e normalizada
        cm_abs = confusion_matrix(y_true, y_pred, labels=classes)
        cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')

        # Prepara anotações combinadas: "Absoluto \n (Percentagem%)"
        anotacoes = np.empty_like(cm_abs, dtype=object)
        for i in range(cm_abs.shape[0]):
            for j in range(cm_abs.shape[1]):
                anotacoes[i, j] = f"{cm_abs[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

        plt.figure(figsize=(9, 7))
        # Tradução das etiquetas para o gráfico
        labels_pt = [c.replace("NEGATIVE", "Negativo").replace("NEUTRAL", "Neutro").replace("POSITIVE", "Positivo") for c in classes]

        sns.heatmap(cm_norm, annot=anotacoes, fmt='', cmap='Blues',
                    xticklabels=labels_pt, yticklabels=labels_pt,
                    vmin=0, vmax=1)

        plt.title('Matriz de Confusão Normalizada: Realidade vs. Previsão', pad=20, fontsize=14)
        plt.xlabel('Previsão do Modelo (RoBERTa)', fontsize=12, labelpad=10)
        plt.ylabel('Classe Real (Gabarito da Persona)', fontsize=12, labelpad=10)
        plt.tight_layout()

        destino = self.reports_dir / "matriz_confusao_normalizada.png"
        plt.savefig(destino, dpi=300)
        plt.close()
        logger.info("Matriz normalizada guardada em: %s", destino)

    def gerar_grafico_barras_metricas(self, y_true: list[str], y_pred: list[str]) -> None:
        """Gera gráfico de barras agrupadas: Precision vs Recall vs F1-Score."""
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Extrai dados apenas das classes principais
        classes = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        labels_pt = ["Negativo", "Neutro", "Positivo"]

        precisions = [report[c]["precision"] for c in classes]
        recalls = [report[c]["recall"] for c in classes]
        f1_scores = [report[c]["f1-score"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
        rects2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
        rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')

        ax.set_ylabel('Pontuação (0.0 a 1.0)', fontsize=12)
        ax.set_title('Desempenho do Modelo por Classe e Métrica', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_pt, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')

        # Adiciona os valores no topo das barras
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        destino = self.reports_dir / "metricas_barras_agrupadas.png"
        plt.savefig(destino, dpi=300)
        plt.close()
        logger.info("Gráfico de barras guardado em: %s", destino)

    def gerar_sankey_diagram(self, y_true: list[str], y_pred: list[str]) -> None:
        """Gera o diagrama de Sankey para mapear o fluxo de erros e acertos."""
        # Criação de índices para os nós do Sankey
        labels = ["Real: Negativo", "Real: Neutro", "Real: Positivo",
                  "Previsto: Negativo", "Previsto: Neutro", "Previsto: Positivo"]

        mapa_nós = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

        # Conta as ligações (Real -> Previsto)
        ligacoes = Counter(zip(y_true, y_pred))

        source = []
        target = []
        value = []
        color_link = []

        # Cores para o fluxo: Verde (Acerto), Vermelho/Cinza (Erro)
        for (real, previsto), contagem in ligacoes.items():
            s_idx = mapa_nós[real]
            t_idx = mapa_nós[previsto] + 3 # +3 para apontar para o lado direito (Previsto)

            source.append(s_idx)
            target.append(t_idx)
            value.append(contagem)

            if real == previsto:
                color_link.append("rgba(46, 204, 113, 0.4)") # Verde transparente
            else:
                color_link.append("rgba(231, 76, 60, 0.4)")  # Vermelho transparente

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20, thickness=30,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#34495e", "#7f8c8d", "#2c3e50", "#34495e", "#7f8c8d", "#2c3e50"]
            ),
            link=dict(
                source=source, target=target, value=value, color=color_link
            )
        )])

        fig.update_layout(title_text="Fluxo de Classificação: Realidade Clínica vs. Previsão do RoBERTa", font_size=12)

        destino = self.reports_dir / "sankey_fluxo_erros.html"
        fig.write_html(str(destino))
        logger.info("Diagrama Sankey guardado em: %s", destino)

    # ==========================================
    # EXECUÇÃO PRINCIPAL
    # ==========================================

    def executar(self) -> None:
        print("A processar métricas e a gerar gráficos...")
        df = self.carregar_resultados()
        y_true, y_pred = self._preparar_dados(df)

        self.exibir_relatorio(y_true, y_pred)

        self.gerar_matriz_normalizada(y_true, y_pred)
        self.gerar_grafico_barras_metricas(y_true, y_pred)
        self.gerar_sankey_diagram(y_true, y_pred)

        print("Processo concluído! Os gráficos foram guardados na pasta 'reports/'.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    AvaliadorDeModelo().executar()
