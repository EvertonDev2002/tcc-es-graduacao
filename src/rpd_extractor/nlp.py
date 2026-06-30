"""
Motor de extração simbólica focado no modelo RPD da
Terapia Cognitivo-Comportamental.
"""

import re
from dataclasses import dataclass, field

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token

from src.configs import LexiconConfig, ModelConfig

# Padrão temporal reutilizável compilado uma única vez no módulo.
_REGEX_TEMPO: re.Pattern[str] = re.compile(
    r"\b(hoje|ontem|amanhã|manhã|tarde|noite|madrugada"
    r"|\d{1,2}h(?:oras?)?|\d{1,2}:\d{2})\b",
    re.IGNORECASE,
)


@dataclass
class DadosRPD:
    """Estrutura intermediária de extração antes da formatação final.

    Usar um dataclass explicita o contrato de dados e elimina dicionários
    com chaves ``str`` opacas trafegando entre métodos privados.
    """

    dia_hora: str = "Não relatado"
    situacoes: list[str] = field(default_factory=list)
    pensamentos: list[str] = field(default_factory=list)
    emocoes_fisicas: set[str] = field(default_factory=set)
    reacoes: list[str] = field(default_factory=list)
    gatilhos: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RegistroRPD:
    """Saída formatada e imutável de uma extração RPD completa."""

    data_hora: str
    situacao: str
    pensamentos: str
    emocao_fisica: str
    reacao: str
    gatilhos: str

    def to_dict(self) -> dict[str, str]:
        """Serializa o registro para um dicionário compatível com pandas."""
        return {
            "Data/Hora": self.data_hora,
            "Situação": self.situacao,
            "Pensamentos": self.pensamentos,
            "Emoção Física": self.emocao_fisica,
            "Reação": self.reacao,
            "Gatilhos": self.gatilhos,
        }


class ExtratorRPD:
    """Motor de extração sintática baseado em regras para o modelo RPD.

    Combina:
    - Expressões regulares para âncoras temporais.
    - ``PhraseMatcher`` do spaCy para detecção de emoções somáticas.
    - Análise de dependências sintáticas para pensamentos e reações.
    """

    def __init__(self) -> None:
        self.nlp: spacy.language.Language = spacy.load(ModelConfig.SPACY_MODEL)
        self._matcher = self._construir_matcher()

    def _construir_matcher(self) -> PhraseMatcher:
        """Compila os padrões do léxico emocional no vocabulário spaCy."""
        matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        for emocao, expressoes in LexiconConfig.LEXICO_EMOCIONAL.items():
            padroes = [self.nlp(texto) for texto in expressoes]
            matcher.add(emocao, padroes)
        return matcher

    # API pública

    def extrair(self, texto: str) -> RegistroRPD:
        """Processa ``texto`` e retorna um :class:`RegistroRPD` estruturado."""
        doc = self.nlp(texto)
        dados = DadosRPD(
            dia_hora=self._extrair_dia_hora(texto),
            emocoes_fisicas=self._extrair_emocoes_fisicas(doc),
        )
        for sent in doc.sents:
            self._processar_sentenca(sent, dados)
        return self._formatar(dados)

    # Extração de âncoras temporais

    @staticmethod
    def _extrair_dia_hora(texto: str) -> str:
        match = _REGEX_TEMPO.search(texto)
        return match.group(0).capitalize() if match else "Não relatado"

    # Detecção somática via PhraseMatcher

    def _extrair_emocoes_fisicas(self, doc: Doc) -> set[str]:
        return {
            self.nlp.vocab.strings[match_id]
            for match_id, _start, _end in self._matcher(doc)
        }

    # Análise por sentença

    def _processar_sentenca(self, sent: Span, dados: DadosRPD) -> None:
        texto_frase = sent.text.strip()
        entidades = [ent.text for ent in sent.ents if ent.label_ in {"PER", "LOC"}]
        tem_contexto = any(
            token.lemma_.lower() in LexiconConfig.SUBSTANTIVOS_CONTEXTO
            for token in sent
        )

        if entidades or tem_contexto:
            dados.situacoes.append(texto_frase)
            dados.gatilhos.extend(entidades)

        for token in sent:
            if token.pos_ in {"VERB", "AUX"}:
                self._analisar_verbo(token, dados)

    def _analisar_verbo(self, token: Token, dados: DadosRPD) -> None:
        lemma = token.lemma_.lower()

        if lemma in LexiconConfig.VERBOS_COGNITIVOS:
            for filho in token.children:
                if filho.dep_ == "ccomp":
                    comp = "".join(t.text_with_ws for t in filho.subtree).strip()
                    if len(comp.split()) > 2:
                        dados.pensamentos.append(comp.capitalize())

        elif lemma in LexiconConfig.VERBOS_ACAO:
            for filho in token.children:
                if filho.dep_ in {"xcomp", "dobj", "obl"}:
                    acao = "".join(t.text_with_ws for t in filho.subtree).strip()
                    if len(acao.split()) > 1:
                        dados.reacoes.append(f"{token.text} {acao}".capitalize())

    # Formatação final

    @staticmethod
    def _formatar(dados: DadosRPD) -> RegistroRPD:
        return RegistroRPD(
            data_hora=dados.dia_hora,
            situacao=(
                " | ".join(
                    dict.fromkeys(dados.situacoes)
                )  # preserva ordem, remove dupl.
                if dados.situacoes
                else "Não descrita claramente."
            ),
            pensamentos=(
                " | ".join(dict.fromkeys(dados.pensamentos))
                if dados.pensamentos
                else "Não verbalizado."
            ),
            emocao_fisica=(
                ", ".join(sorted(dados.emocoes_fisicas))
                if dados.emocoes_fisicas
                else "Falta léxico."
            ),
            reacao=(
                " | ".join(dict.fromkeys(dados.reacoes))
                if dados.reacoes
                else "Sem reação."
            ),
            gatilhos=", ".join(dict.fromkeys(dados.gatilhos)),
        )
