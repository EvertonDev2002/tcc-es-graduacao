"""Avaliação estatística do classificador RoBERTa no corpus RPD.

Este módulo calcula as métricas de desempenho computacional descritas
na seção 4.5.1 do TCC (acurácia, precision, recall, f1-score) e gera
a Matriz de Confusão normalizada (Figura 4 do TCC).

Metodologia do gabarito (y_true):
    Como o corpus é sintético, não há anotação humana por registro.
    O gabarito é definido pela trajetória emocional *esperada* de cada
    persona, conforme a Tabela 2 do TCC. Esta é uma aproximação válida
    para uma PoC — a validação real por especialista humano (Kappa de
    Cohen) está planejada para o TCC 2.

    Persona A → POSITIVE  (melhora gradual)
    Persona B → NEGATIVE  (crises de ansiedade)
    Persona C → NEGATIVE  (declínio progressivo)
    Persona D → NEUTRAL   (alexitimia severa — foco em eventos externos)
    Persona E → NEGATIVE  (perfil complexo, oscilações com tendência negativa)

Uso:
    uv run python scripts/avaliar_modelo.py
"""

import logging
from pathlib import Path
from typing import ClassVar, Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# Constantes

# Gabarito de polaridade por persona, baseado na Tabela 2 do TCC (seção 4.3).
# Representa a trajetória emocional *dominante* esperada, não a polaridade
# de cada registro individual.
_GABARITO_PERSONAS: Final[dict[str, str]] = {
    "Persona_A": "POSITIVE",
    "Persona_B": "NEGATIVE",
    "Persona_C": "NEGATIVE",
    "Persona_D": "NEUTRAL",
    "Persona_E": "NEGATIVE",
}

# Ordem fixa das classes para garantir consistência na matriz de confusão
_CLASSES_ORDENADAS: Final[list[str]] = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Avaliador


class AvaliadorDeModelo:
    """Calcula e apresenta métricas de avaliação do classificador RoBERTa.

    Separa claramente três responsabilidades:
    - Carregamento e preparação dos dados
    - Cálculo e exibição de métricas textuais
    - Geração e persistência da matriz de confusão
    """

    GABARITO: ClassVar[Final[dict[str, str]]] = _GABARITO_PERSONAS

    def __init__(self, caminho_csv: Path | None = None) -> None:
        self._caminho_csv = caminho_csv or PathConfig.CORPUS_PROCESSED_PATH

    # Carregamento e preparação

    def carregar_resultados(self) -> pd.DataFrame:
        """Lê o CSV processado pelo pipeline."""
        if not self._caminho_csv.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self._caminho_csv}. "
                "Execute o pipeline (run_pipeline.py) antes de avaliar."
            )
        logger.info("Carregando resultados de: %s", self._caminho_csv)
        return pd.read_csv(self._caminho_csv)

    def preparar_dados(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Constrói os vetores y_true e y_pred a partir do gabarito por persona.

        Args:
            df: DataFrame com colunas "persona" e "Polaridade (RoBERTa)".

        Returns:
            Tupla (y_true, y_pred) com as listas de rótulos esperados e preditos.

        Note:
            Registros cujo persona não consta no gabarito são descartados
            com aviso de log — evita falha silenciosa.
        """
        df = df.copy()
        df["Polaridade_Esperada"] = df["persona"].map(self.GABARITO)

        n_sem_gabarito = df["Polaridade_Esperada"].isna().sum()
        if n_sem_gabarito > 0:
            personas_desconhecidas = df.loc[
                df["Polaridade_Esperada"].isna(), "persona"
            ].unique()
            logger.warning(
                "%d registros sem gabarito descartados. Personas desconhecidas: %s",
                n_sem_gabarito,
                list(personas_desconhecidas),
            )

        dados_validos = df.dropna(subset=["Polaridade_Esperada", "Polaridade (RoBERTa)"])
        y_true: list[str] = dados_validos["Polaridade_Esperada"].tolist()
        y_pred: list[str] = dados_validos["Polaridade (RoBERTa)"].tolist()

        logger.info(
            "Vetores preparados: %d registros válidos de %d totais.",
            len(y_true),
            len(df),
        )
        return y_true, y_pred

    # Métricas textuais

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
        print(f"  Precisão (Macro) : {precision:.4f}")
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

    # Matriz de confusão

    def _calcular_matriz(
        self, y_true: list[str], y_pred: list[str]
    ) -> tuple[object, list[str]]:
        """Calcula a matriz de confusão com classes na ordem canônica."""
        classes_presentes = [c for c in _CLASSES_ORDENADAS if c in set(y_true + y_pred)]
        cm = confusion_matrix(y_true, y_pred, labels=classes_presentes)
        return cm, classes_presentes

    def _plotar_matriz(self, cm: object, classes: list[str]) -> plt.Figure:
        """Gera a figura do heatmap da matriz de confusão."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title("Matriz de Confusão: Predição RoBERTa vs Gabarito por Persona")
        ax.set_xlabel("Predição do Modelo (RoBERTa)")
        ax.set_ylabel("Classe Esperada (Gabarito)")
        fig.tight_layout()
        return fig

    def _salvar_figura(self, fig: plt.Figure, destino: Path) -> None:
        """Persiste a figura em disco e registra o caminho no log."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=150)
        plt.close(fig)
        logger.info("Matriz de confusão salva em: %s", destino)
        print(f"Gráfico salvo em: {destino}")

    def gerar_matriz_confusao(
        self,
        y_true: list[str],
        y_pred: list[str],
        destino: Path | None = None,
    ) -> None:
        """Gera e salva o heatmap da Matriz de Confusão.

        Args:
            y_true: Rótulos esperados.
            y_pred: Rótulos preditos pelo RoBERTa.
            destino: Caminho de saída. Padrão: ``reports/matriz_confusao_RoBERTa.png``.
        """
        caminho = (
            destino or PathConfig.BASE_DIR / "reports" / "matriz_confusao_RoBERTa.png"
        )
        cm, classes = self._calcular_matriz(y_true, y_pred)
        fig = self._plotar_matriz(cm, classes)
        self._salvar_figura(fig, caminho)

    # Orquestrador

    def executar(self) -> None:
        """Executa o fluxo completo de avaliação."""
        df = self.carregar_resultados()
        y_true, y_pred = self.preparar_dados(df)
        self.exibir_relatorio(y_true, y_pred)
        self.gerar_matriz_confusao(y_true, y_pred)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    AvaliadorDeModelo().executar()
