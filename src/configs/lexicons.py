"""Dicionários de termos clínicos, contextos situacionais e listas de exclusão gramatical."""

from typing import ClassVar, Final


class LexiconConfig:
    """Léxicos e stopwords utilizados na extração simbólica do RPD.

    Todos os atributos são somente-leitura. Use ``frozenset`` para conjuntos
    e ``tuple`` para sequências que não devem ser mutadas em runtime.
    """

    # --- Stopwords clínicas (palavras de alta frequência sem valor semântico) ---
    STOPWORDS_CLINICAS: ClassVar[Final[frozenset[str]]] = frozenset(
        {
            "hoje",
            "dia",
            "mesmo",
            "hora",
            "coisa",
            "vez",
            "pouco",
            "nada",
            "tudo",
            "ano",
        }
    )

    # --- Substantivos âncora de situações clínicas relevantes ---
    SUBSTANTIVOS_CONTEXTO: ClassVar[Final[frozenset[str]]] = frozenset(
        {
            "trabalho",
            "casa",
            "rua",
            "mercado",
            "reunião",
            "chefe",
            "escola",
            "faculdade",
            "ônibus",
            "carro",
            "projeto",
            "parque",
            "amiga",
            "amigo",
            "esposa",
        }
    )

    # --- Léxico emocional-somático mapeado por categoria clínica ---
    LEXICO_EMOCIONAL: ClassVar[Final[dict[str, list[str]]]] = {
        "Ansiedade": [
            "nó no estômago",
            "coração acelerar",
            "suor frio",
            "mão suar",
            "ansioso",
            "ansiedade",
            "pânico",
        ],
        "Angústia": [
            "aperto no peito",
            "peso nas costas",
            "vontade de sumir",
            "tristeza",
            "buraco no peito",
            "triste",
        ],
        "Raiva": [
            "sangue ferver",
            "ficar cego",
            "corpo quente",
            "ódio",
            "irritado",
            "raiva",
        ],
        "Exaustão": [
            "bateria acabar",
            "corpo pesado",
            "mente nublado",
            "exausto",
            "apagão mental",
            "cansado",
        ],
    }

    # --- Verbos cognitivos (indicam pensamentos automáticos) ---
    VERBOS_COGNITIVOS: ClassVar[Final[frozenset[str]]] = frozenset(
        {
            "pensar",
            "achar",
            "imaginar",
            "acreditar",
            "concluir",
            "ter",
            "saber",
            "sentir",
        }
    )

    # --- Verbos de ação (indicam reações comportamentais) ---
    VERBOS_ACAO: ClassVar[Final[frozenset[str]]] = frozenset(
        {
            "fazer",
            "ir",
            "parar",
            "cancelar",
            "sair",
            "trancar",
            "bater",
            "evitar",
            "chorar",
            "voltar",
        }
    )
