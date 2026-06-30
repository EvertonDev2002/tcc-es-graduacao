"""Gerador automatizado de corpus sintético de diários clínicos via APIs de LLMs."""

import csv
import logging
import time
from pathlib import Path

from google import genai
from mistralai.client import Mistral

from src.configs import (
    GeneratorConfig,
    GeneratorSettings,
    PathConfig,
    PersonaConfig,
)

logger = logging.getLogger(__name__)


class GeradorCorpus:
    """Motor de geração automatizada de dados sintéticos via APIs de LLMs.

    Suporta Gemini (Google) e Mistral AI como backends intercambiáveis,
    selecionados por persona conforme configurado em :class:`GeneratorConfig`.
    """

    def __init__(self, settings: GeneratorSettings | None = None) -> None:
        self._settings = settings or GeneratorSettings()
        self._settings.validate_keys()

        self._gemini_client = genai.Client(
            api_key=self._settings.gemini_api_key.get_secret_value()
        )
        self._mistral_client = Mistral(
            api_key=self._settings.mistral_api_key.get_secret_value()
        )
        PathConfig.RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _chamar_gemini(self, prompt: str) -> str:
        resposta = self._gemini_client.models.generate_content(
            model=GeneratorConfig.GEMINI_MODEL,
            contents=prompt,
        )
        return resposta.text

    def _chamar_mistral(self, prompt: str) -> str:
        resposta = self._mistral_client.chat.complete(
            model=GeneratorConfig.MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resposta.choices[0].message.content

    def _gerar_texto(self, persona: PersonaConfig, prompt: str) -> str:
        """Despacha o prompt para o backend correto conforme ``persona.motor_ia``."""
        if persona.motor_ia == "gemini":
            return self._chamar_gemini(prompt)
        if persona.motor_ia == "mistral":
            return self._chamar_mistral(prompt)
        raise ValueError(f"Backend desconhecido: '{persona.motor_ia}'")

    # Pipeline principal

    def executar(self, caminho_saida: Path | None = None) -> None:
        """Gera o corpus completo e salva em ``caminho_saida``.

        Args:
            caminho_saida: Destino do CSV. Usa :attr:`PathConfig.CORPUS_RAW_PATH`
                se não informado.
        """
        destino = caminho_saida or PathConfig.CORPUS_RAW_PATH
        logger.info("Gerando corpus em: %s", destino)

        with destino.open(mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["id_registro", "persona", "texto_diario"])
            self._iterar_personas(writer)

        logger.info("Corpus finalizado: %s", destino)

    def _iterar_personas(self, writer: csv.writer) -> None:  # type: ignore[type-arg]
        id_registro = 1
        for persona in GeneratorConfig.PERSONAS:
            logger.info(
                "Iniciando requisições: %s via %s",
                persona.id,
                persona.motor_ia.upper(),
            )
            id_registro = self._gerar_dias(writer, persona, id_registro)

    def _gerar_dias(
        self,
        writer: csv.writer,  # type: ignore[type-arg]
        persona: PersonaConfig,
        id_registro: int,
    ) -> int:
        total = GeneratorConfig.DIAS_MONITORAMENTO
        for dia in range(1, total + 1):
            prompt = (
                f"Escreva o dia {dia}/{total} do diário de: {persona.prompt_base}. "
                f"Requisito: Retorne estritamente o texto narrativo em primeira pessoa, "
                f"com pelo menos 150 palavras, citando locais e pessoas. Sem aspas ou prefixos."
            )
            try:
                texto = self._gerar_texto(persona, prompt)
                writer.writerow([id_registro, persona.id, texto.strip()])
                id_registro += 1
                time.sleep(GeneratorConfig.TEMPO_ESPERA_API)
            except Exception:
                logger.exception(
                    "Erro no dia %d da persona '%s'. Registro ignorado.",
                    dia,
                    persona.id,
                )
        return id_registro


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    GeradorCorpus().executar()
