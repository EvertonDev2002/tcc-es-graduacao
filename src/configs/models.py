"""Especificações de modelos neurais e parâmetros de pipelines de PLN."""

from typing import ClassVar, Final


class ModelConfig:
    """Identificadores de modelos de redes neurais e pipelines spaCy."""

    BERT_MODEL: ClassVar[Final[str]] = "Adilmar/caramelo-smile-2"
    SPACY_MODEL: ClassVar[Final[str]] = "pt_core_news_lg"


class PipelineParameters:
    """Hiperparâmetros dos algoritmos de janelamento e sobreposição de texto."""

    WINDOW_SIZE: ClassVar[Final[int]] = 400
    OVERLAP: ClassVar[Final[int]] = 50
