"""
Orquestrador do pipeline de inferência neural
e extração simbólica do RPD.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.configs import (
    LexiconConfig,
    ModelConfig,
    PathConfig,
    PipelineParameters,
)
from src.rpd_extractor import ExtratorRPD

logger = logging.getLogger(__name__)

# POS tags retidas na nuvem de palavras clínica
_POS_NUVEM: frozenset[str] = frozenset({"NOUN", "ADJ", "ADV"})

# Sliding Window


@dataclass(frozen=True)
class ResultadoJanela:
    """Saída bruta de uma única janela de inferência.

    Attributes:
        logits: Vetor de logits crus (antes do softmax), shape (n_classes,).
        tokens_inicio: Índice do primeiro token desta janela no texto original.
        tokens_fim: Índice do último token desta janela no texto original.
    """

    logits: np.ndarray
    tokens_inicio: int
    tokens_fim: int


class SlidingWindowClassificador:
    """Classifica textos longos via janelamento deslizante com agregação por média de logits.

    O RoBERTa tem limite de 512 tokens. Textos de diários terapêuticos frequentemente
    excedem esse limite. Esta classe resolve o problema segmentando o texto em janelas
    sobrepostas e agregando os resultados.

    Estratégia de agregação (Sun et al., 2019):
        1. Tokeniza o texto completo sem truncamento.
        2. Segmenta os token ids em janelas de tamanho ``window_size`` com
           sobreposição de ``overlap`` tokens.
        3. Roda inferência RoBERTa em cada janela individualmente.
        4. Calcula a média aritmética dos logits de todas as janelas.
        5. Aplica argmax sobre a média — a classe com maior logit médio vence.

    Por que média de logits e não majority voting?
        O logit carrega a *intensidade* da predição. Uma janela com logit
        NEG=2.1 contribui mais do que uma com NEG=0.1, mesmo que ambas
        resultem na classe NEGATIVE. O majority voting perde essa informação.

    Args:
        modelo: Identificador HuggingFace do modelo (ex: "Adilmar/caramelo-smile-2").
        device: Dispositivo PyTorch ("cuda" ou "cpu").
        window_size: Número de tokens por janela (padrão: 400, conforme TCC seção 4.4.2).
        overlap: Sobreposição entre janelas consecutivas (padrão: 50).

    Note:
        ``window_size`` + tokens especiais ([CLS], [SEP]) deve ser <= 512.
        Com window_size=400 e 2 tokens especiais, ficamos em 402 — dentro do limite.
    """

    # Tokens especiais [CLS] e [SEP] que o tokenizador insere automaticamente
    _TOKENS_ESPECIAIS: int = 2

    def __init__(
        self,
        modelo: str,
        device: torch.device,
        window_size: int = PipelineParameters.WINDOW_SIZE,
        overlap: int = PipelineParameters.OVERLAP,
    ) -> None:
        if window_size + self._TOKENS_ESPECIAIS > 512:
            raise ValueError(
                f"window_size={window_size} + {self._TOKENS_ESPECIAIS} tokens especiais "
                f"excede o limite de 512 do RoBERTa."
            )
        if overlap >= window_size:
            raise ValueError(
                f"overlap={overlap} deve ser menor que window_size={window_size}."
            )

        self._device = device
        self._window_size = window_size
        self._overlap = overlap
        self._stride = window_size - overlap  # passo entre janelas consecutivas

        self._tokenizer = AutoTokenizer.from_pretrained(modelo)
        self._model = AutoModelForSequenceClassification.from_pretrained(modelo).to(
            device
        )
        self._model.eval()
        self._labels: dict[int, str] = self._model.config.id2label

        logger.info(
            "SlidingWindowClassificador inicializado | modelo=%s | "
            "window=%d | overlap=%d | stride=%d | device=%s",
            modelo,
            window_size,
            overlap,
            self._stride,
            device,
        )

    # API pública

    def classificar(self, texto: str) -> str:
        """Classifica ``texto`` e retorna a classe de polaridade como string.

        Para textos curtos (≤ window_size tokens), equivale a uma única
        inferência RoBERTa convencional. Para textos longos, aplica o
        janelamento deslizante com agregação por média de logits.

        Returns:
            Rótulo de polaridade em maiúsculas: "POSITIVE", "NEUTRAL" ou "NEGATIVE".
        """
        token_ids = self._tokenizar_completo(texto)

        if len(token_ids) <= self._window_size:
            # Texto curto: inferência direta, sem janelamento
            logits = self._inferir_janela(token_ids)
            logger.debug("Texto curto (%d tokens): inferência direta.", len(token_ids))
        else:
            logits = self._classificar_com_janelamento(token_ids)

        classe_idx = int(np.argmax(logits))
        return self._labels.get(classe_idx, "NEUTRAL").upper()

    def classificar_com_detalhes(self, texto: str) -> dict[str, object]:
        """Versão estendida que retorna logits e número de janelas processadas.

        Útil para depuração, validação e para o cálculo do Kappa de Cohen,
        pois expõe a confiança do modelo em cada classe.

        Returns:
            Dicionário com:
            - ``classe``: rótulo predito (str)
            - ``logits_medios``: array com logit médio por classe (np.ndarray)
            - ``probabilidades``: softmax dos logits médios (np.ndarray)
            - ``n_janelas``: número de janelas processadas (int)
            - ``confianca``: probabilidade da classe predita (float)
        """
        token_ids = self._tokenizar_completo(texto)
        n_janelas: int

        if len(token_ids) <= self._window_size:
            logits_medios = self._inferir_janela(token_ids)
            n_janelas = 1
        else:
            logits_medios = self._classificar_com_janelamento(token_ids)
            n_janelas = self._contar_janelas(len(token_ids))

        classe_idx = int(np.argmax(logits_medios))
        probs = self._softmax(logits_medios)

        return {
            "classe": self._labels.get(classe_idx, "NEUTRAL").upper(),
            "logits_medios": logits_medios,
            "probabilidades": probs,
            "n_janelas": n_janelas,
            "confianca": float(probs[classe_idx]),
        }

    # Tokenização

    def _tokenizar_completo(self, texto: str) -> list[int]:
        """Tokeniza sem truncamento e sem tokens especiais.

        Os tokens especiais ([CLS], [SEP]) são adicionados por janela
        em ``_montar_input_janela``, não aqui — evitar duplicação.
        """
        return self._tokenizer.encode(
            str(texto),
            add_special_tokens=False,  # adicionados por janela
            truncation=False,  # sem truncamento — este é o ponto central
        )

    # Janelamento

    def _classificar_com_janelamento(self, token_ids: list[int]) -> np.ndarray:
        """Aplica janelamento deslizante e retorna a média dos logits.

        Fluxo:
            token_ids → janelas → inferência por janela → stack → média

        Args:
            token_ids: Lista de token ids sem tokens especiais.

        Returns:
            Array com logit médio por classe, shape (n_classes,).
        """
        janelas = self._segmentar(token_ids)
        logger.debug(
            "Texto longo (%d tokens): %d janelas geradas (stride=%d).",
            len(token_ids),
            len(janelas),
            self._stride,
        )

        logits_por_janela: list[np.ndarray] = []
        for janela in janelas:
            logits = self._inferir_janela(
                janela.logits
            )  # janela.logits = token_ids da janela
            logits_por_janela.append(logits)

        # Stack shape: (n_janelas, n_classes) → média ao longo do eixo 0
        return np.mean(np.stack(logits_por_janela, axis=0), axis=0)

    def _segmentar(self, token_ids: list[int]) -> list[ResultadoJanela]:
        """Segmenta ``token_ids`` em janelas sobrepostas.

        Cada janela é representada como ``ResultadoJanela`` com os token ids
        armazenados temporariamente no campo ``logits`` (reutilizado como
        portador de dados antes da inferência).

        Exemplo com window=5, overlap=2, stride=3, tokens=[0..11]:
            Janela 0: tokens[0:5]   → índices 0–4
            Janela 1: tokens[3:8]   → índices 3–7
            Janela 2: tokens[6:11]  → índices 6–10
            Janela 3: tokens[9:12]  → índices 9–11  (última, menor que window_size)

        Note:
            A última janela pode ser menor que ``window_size``. Isso é
            intencional — garantir cobertura total do texto.
        """
        janelas: list[ResultadoJanela] = []
        inicio = 0
        total = len(token_ids)

        while inicio < total:
            fim = min(inicio + self._window_size, total)
            ids_janela = token_ids[inicio:fim]
            janelas.append(
                ResultadoJanela(
                    logits=np.array(ids_janela),  # token ids temporários
                    tokens_inicio=inicio,
                    tokens_fim=fim,
                )
            )
            if fim == total:
                break
            inicio += self._stride

        return janelas

    def _contar_janelas(self, n_tokens: int) -> int:
        """Calcula o número de janelas sem segmentar de fato (útil para logs)."""
        if n_tokens <= self._window_size:
            return 1
        return 1 + max(0, -(-(n_tokens - self._window_size) // self._stride))

    # Inferência

    def _inferir_janela(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        """Roda o RoBERTa em uma única janela de token ids.

        Args:
            token_ids: Token ids *sem* tokens especiais.

        Returns:
            Logits crus, shape (n_classes,).
        """
        input_ids = self._montar_input_janela(list(token_ids))
        tensor = torch.tensor([input_ids], dtype=torch.long).to(self._device)

        with torch.no_grad():
            saida = self._model(input_ids=tensor)

        return saida.logits.cpu().numpy()[0]

    def _montar_input_janela(self, token_ids: list[int]) -> list[int]:
        """Adiciona <s> no início e <s> no fim, como o RoBERTa espera."""
        cls_id: int = self._tokenizer.cls_token_id  # type: ignore[assignment]
        sep_id: int = self._tokenizer.sep_token_id  # type: ignore[assignment]
        return [cls_id, *token_ids, sep_id]

    # Utilitários

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax numericamente estável (subtrai o máximo antes de exp)."""
        e = np.exp(logits - np.max(logits))
        return e / e.sum()


# Extrator de nuvem de palavras


class ExtratorNuvem:
    """Filtra lemas semanticamente relevantes para a nuvem de palavras clínica."""

    def __init__(self, extrator_rpd: ExtratorRPD) -> None:
        self._nlp = extrator_rpd.nlp

    def extrair(self, texto: str) -> str:
        """Retorna lemas filtrados como string separada por espaços."""
        doc = self._nlp(texto)
        return " ".join(
            token.lemma_.lower()
            for token in doc
            if token.pos_ in _POS_NUVEM
            and not token.is_stop
            and len(token.lemma_) > 2
            and token.lemma_.lower() not in LexiconConfig.STOPWORDS_CLINICAS
        )


# Orquestrador do pipeline


class PipelineOrquestrador:
    """Gerencia e executa inferências neurais e extrações simbólicas do corpus RPD.

    Args:
        entrada: Caminho do CSV bruto. Padrão: :attr:`PathConfig.CORPUS_RAW_PATH`.
        saida: Caminho do CSV processado. Padrão: :attr:`PathConfig.CORPUS_PROCESSED_PATH`.
    """

    def __init__(
        self,
        entrada: Path | None = None,
        saida: Path | None = None,
    ) -> None:
        self._entrada = entrada or PathConfig.CORPUS_RAW_PATH
        self._saida = saida or PathConfig.CORPUS_PROCESSED_PATH
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Dispositivo de inferência: %s", self._device)

        self._extrator_rpd = ExtratorRPD()
        self._classificador = SlidingWindowClassificador(
            ModelConfig.BERT_MODEL,
            self._device,
        )
        self._extrator_nuvem = ExtratorNuvem(self._extrator_rpd)

    def executar(self) -> None:
        """Lê o corpus bruto, processa cada entrada e salva o CSV enriquecido."""
        df = self._carregar_dados()
        resultados = self._processar(df)
        self._salvar(df, resultados)

    def _carregar_dados(self) -> pd.DataFrame:
        logger.info("Lendo corpus de: %s", self._entrada)
        return pd.read_csv(self._entrada, engine="python")

    def _processar(self, df: pd.DataFrame) -> list[dict[str, str]]:
        resultados: list[dict[str, str]] = []

        for _idx, linha in tqdm(df.iterrows(), total=len(df), desc="Pipeline RPD"):
            texto = str(linha["texto_diario"])
            registro = self._extrator_rpd.extrair(texto)
            resultado = registro.to_dict()
            resultado["Polaridade (RoBERTa)"] = self._classificador.classificar(texto)
            resultado["Palavras_Nuvem"] = self._extrator_nuvem.extrair(texto)
            resultados.append(resultado)

        return resultados

    def _salvar(self, df: pd.DataFrame, resultados: list[dict[str, str]]) -> None:
        df_final = pd.concat([df, pd.DataFrame(resultados)], axis=1)
        PathConfig.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(self._saida, index=False, encoding="utf-8-sig")
        logger.info("Corpus processado salvo em: %s", self._saida)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    PipelineOrquestrador().executar()
