"""Configurações e entidades de domínio para geração de corpus sintético via LLMs."""

from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class PersonaConfig:
    """Representação imutável de uma persona clínica para modelagem sintética.

    Attributes:
        id: Identificador único da persona (ex.: "Persona_A").
        motor_ia: Backend de geração ("gemini" | "mistral").
        prompt_base: Descrição clínica usada como contexto nos prompts.
    """

    id: str
    motor_ia: str
    prompt_base: str


class GeneratorSettings(BaseSettings):
    """Segredos e parâmetros carregados do ambiente (.env ou variáveis de sistema).

    Nunca armazene chaves de API como strings literais no código-fonte.
    Use um arquivo ``.env`` local (excluído do VCS via ``.gitignore``)
    ou injete as variáveis diretamente no ambiente de execução.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr = SecretStr("")
    mistral_api_key: SecretStr = SecretStr("")

    def validate_keys(self) -> None:
        """Lança ``ValueError`` se alguma chave obrigatória estiver ausente."""
        missing = [
            name
            for name, val in (
                ("GEMINI_API_KEY", self.gemini_api_key),
                ("MISTRAL_API_KEY", self.mistral_api_key),
            )
            if not val.get_secret_value()
        ]
        if missing:
            raise ValueError(
                f"Chaves de API ausentes no ambiente: {', '.join(missing)}"
            )


# Personas definidas como constante de módulo (não como atributo de classe)
# para evitar o antipadrão de lista mutável em ClassVar.
_PERSONAS: Final[list[PersonaConfig]] = [
    PersonaConfig(
        id="Persona_A",
        motor_ia="gemini",
        prompt_base=(
            "Mariana, 22 anos, Fortaleza. "
            "Melhora gradual. Vocabulário emocional aumenta ao longo dos dias."
        ),
    ),
    PersonaConfig(
        id="Persona_B",
        motor_ia="gemini",
        prompt_base="Sheldon, 35 anos, São Paulo. Estável com crises isoladas de ansiedade.",
    ),
    PersonaConfig(
        id="Persona_C",
        motor_ia="gemini",
        prompt_base="Letícia, 17 anos, Russas. Declínio. Frases curtas, desamparo crescente.",
    ),
    PersonaConfig(
        id="Persona_D",
        motor_ia="mistral",
        prompt_base=(
            "Jorge, 52 anos, Curitiba. "
            "Alexitimia severa. Descreve apenas fatos, rotina e "
            "sensações físicas no lugar de emoções."
        ),
    ),
    PersonaConfig(
        id="Persona_E",
        motor_ia="mistral",
        prompt_base=(
            "Alex, 28 anos, Porto Alegre. "
            "TDAH e autismo. Mudanças abruptas de tópico, "
            "relatos de exaustão social extrema (masking)."
        ),
    ),
]


class GeneratorConfig:
    """Parâmetros operacionais estáticos do pipeline de geração de corpus."""

    GEMINI_MODEL: ClassVar[Final[str]] = "gemini-3.1-flash-lite"
    MISTRAL_MODEL: ClassVar[Final[str]] = "mistral-medium-3-5"

    DIAS_MONITORAMENTO: ClassVar[Final[int]] = 60
    TEMPO_ESPERA_API: ClassVar[Final[int]] = 2

    PERSONAS: ClassVar[Final[list[PersonaConfig]]] = _PERSONAS
