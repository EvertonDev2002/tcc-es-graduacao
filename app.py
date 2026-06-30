"""Dashboard clínico Streamlit — Rastreio Emocional Automatizado (RPD)."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src.configs.paths import PathConfig

# Constantes de apresentação

_CSV_PATH = PathConfig.CORPUS_PROCESSED_PATH

_COLORSCALE_HUMOR = [
    [0.0, "#E74C3C"],
    [0.5, "#BDC3C7"],
    [1.0, "#2ECC71"],
]

_MAPA_POLARIDADE: dict[str, int] = {
    "POSITIVE": 1,
    "NEUTRAL": 0,
    "NEGATIVE": -1,
}

# Larguras das colunas da tabela RPD em pixels
_LARGURAS_COLUNAS_RPD: dict[str, int] = {
    "Data/Hora": 120,
    "Situação": 280,
    "Pensamento Automático": 280,
    "Emoção e Somatização": 220,
    "Reação / Comportamento": 220,
}

# Camada de dados


@st.cache_data
def carregar_dados(caminho: str = str(_CSV_PATH)) -> pd.DataFrame:
    """Carrega e enriquece o CSV com colunas de sequência temporal."""
    df = pd.read_csv(caminho)
    df["Dia"] = df.groupby("persona").cumcount() + 1
    return df


def preparar_persona(df: pd.DataFrame, persona: str) -> pd.DataFrame:
    """Filtra e enriquece as colunas de calendário para uma persona específica."""
    df_p = df.loc[df["persona"] == persona].copy()
    df_p["Semana"] = ((df_p["Dia"] - 1) // 7) + 1
    df_p["Dia_Semana"] = ((df_p["Dia"] - 1) % 7) + 1
    df_p["Valor_Polaridade"] = df_p["Polaridade (RoBERTa)"].map(_MAPA_POLARIDADE)
    return df_p


# Componentes visuais


def _grafico_radar_somatizacao(df_persona: pd.DataFrame) -> None:
    df_emocoes = df_persona[df_persona["Emoção Física"] != "Falta léxico."]

    if df_emocoes.empty:
        st.info("Dados de somatização insuficientes para esta persona.")
        return

    contagem = (
        df_emocoes["Emoção Física"]
        .str.split(", ")
        .explode()
        .value_counts()
        .reset_index()
    )
    contagem.columns = ["Emocao", "Frequencia"]

    fig = go.Figure(
        go.Scatterpolar(
            r=contagem["Frequencia"],
            theta=contagem["Emocao"],
            fill="toself",
            line={"color": "#2E86C1", "width": 2},
            fillcolor="rgba(46,134,193,0.25)",
        )
    )
    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "tickfont": {"size": 10},
                "gridcolor": "rgba(255,255,255,0.15)",
            },
            "angularaxis": {"tickfont": {"size": 11}},
            "bgcolor": "rgba(0,0,0,0)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin={"l": 50, "r": 50, "t": 20, "b": 20},
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


def _grafico_mapa_calor(df_persona: pd.DataFrame) -> None:
    polaridade_label = df_persona["Polaridade (RoBERTa)"].map(
        {"POSITIVE": "Positivo", "NEUTRAL": "Neutro", "NEGATIVE": "Negativo"}
    )

    fig = go.Figure(
        go.Heatmap(
            z=df_persona["Valor_Polaridade"],
            x=df_persona["Dia_Semana"],
            y=df_persona["Semana"],
            colorscale=_COLORSCALE_HUMOR,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar={
                "title": "Polaridade",
                "tickvals": [-1, 0, 1],
                "ticktext": ["Negativo", "Neutro", "Positivo"],
                "tickfont": {"size": 11},
                "len": 0.8,
            },
            text=polaridade_label,
            hovertemplate=(
                "Semana %{y} · Dia %{x}<br>Polaridade: %{text}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis={
            "title": "Dia da Semana",
            "dtick": 1,
            "tickvals": list(range(1, 8)),
            "ticktext": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
            "tickfont": {"size": 11},
        },
        yaxis={
            "title": "Semana",
            "dtick": 1,
            "autorange": "reversed",
            "tickfont": {"size": 11},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 60, "r": 20, "t": 20, "b": 50},
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


def _grafico_evolucao_polaridade(df_persona: pd.DataFrame) -> None:
    """Linha temporal da polaridade ao longo dos dias de monitoramento."""
    cor_map = {"POSITIVE": "#2ECC71", "NEUTRAL": "#BDC3C7", "NEGATIVE": "#E74C3C"}

    fig = go.Figure()

    # Linha de tendência contínua
    fig.add_trace(
        go.Scatter(
            x=df_persona["Dia"],
            y=df_persona["Valor_Polaridade"],
            mode="lines",
            line={"color": "rgba(150,150,150,0.35)", "width": 1.5},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Pontos coloridos por classe
    for polaridade, valor in _MAPA_POLARIDADE.items():
        mask = df_persona["Polaridade (RoBERTa)"] == polaridade
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df_persona.loc[mask, "Dia"],
                    y=df_persona.loc[mask, "Valor_Polaridade"],
                    mode="markers",
                    name=polaridade.capitalize(),
                    marker={
                        "color": cor_map[polaridade],
                        "size": 8,
                        "line": {"width": 1, "color": "rgba(255,255,255,0.4)"},
                    },
                    hovertemplate=f"Dia %{{x}}<br>{polaridade}<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis={"title": "Dia de Monitoramento", "tickfont": {"size": 11}},
        yaxis={
            "title": "Polaridade",
            "tickvals": [-1, 0, 1],
            "ticktext": ["Negativo", "Neutro", "Positivo"],
            "tickfont": {"size": 11},
            "gridcolor": "rgba(255,255,255,0.08)",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 80, "r": 20, "t": 30, "b": 50},
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def _nuvem_palavras(df_persona: pd.DataFrame) -> None:
    texto = " ".join(df_persona["Palavras_Nuvem"].dropna())
    if not texto.strip():
        st.info("Palavras-chave insuficientes para esta persona.")
        return

    wc = WordCloud(
        width=900,
        height=420,
        background_color=None,  # transparente
        mode="RGBA",
        colormap="RdGy",
        max_words=70,
        max_font_size=96,
        min_font_size=12,
        collocations=False,  # evita bigramas repetidos
        prefer_horizontal=0.85,
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _card_rpd(label: str, conteudo: str) -> None:
    """Renderiza um campo RPD como card com markdown dentro de um container."""
    with st.container(border=True):
        st.markdown(f"**{label}**")
        texto = str(conteudo).replace(" | ", " **📌** ")
        st.markdown(str(texto))


def _tabela_rpd(df_persona: pd.DataFrame, persona: str) -> None:
    dia = st.slider(
        "Dia de monitoramento:",
        min_value=1,
        max_value=len(df_persona),
        value=1,
        key=f"slider_rpd_{persona}",
    )
    registro = df_persona.loc[df_persona["Dia"] == dia].iloc[0]

    # Cabeçalho: polaridade + âncora temporal
    polaridade = registro["Polaridade (RoBERTa)"]
    icone = {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}.get(polaridade, "⚪")
    with st.container(border=True):
        col_meta_a, col_meta_b = st.columns(2)
        col_meta_a.markdown(f"**{icone} Polaridade:** {polaridade}")
        col_meta_b.markdown(f"**🕐 Âncora temporal:** {registro['Data/Hora']}")

    # Campos RPD em duas colunas
    col_a, col_b = st.columns(2)
    with col_a:
        _card_rpd("📍 Situação", registro["Situação"])
        _card_rpd("😣 Emoção Física", registro["Emoção Física"])
    with col_b:
        _card_rpd("💭 Pensamento Automático", registro["Pensamentos"])
        _card_rpd("⚡ Reação / Comportamento", registro["Reação"])

    with st.expander("📄 Relato integral do paciente"):
        st.markdown(str(registro["texto_diario"]))


def _metricas_resumo(df_persona: pd.DataFrame) -> None:
    """Cartões de resumo no topo da aba."""
    total = len(df_persona)
    pos = (df_persona["Polaridade (RoBERTa)"] == "POSITIVE").sum()
    neg = (df_persona["Polaridade (RoBERTa)"] == "NEGATIVE").sum()
    neu = (df_persona["Polaridade (RoBERTa)"] == "NEUTRAL").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dias monitorados", total)
    c2.metric("Registros positivos", pos, f"{pos / total:.0%}")
    c3.metric("Registros negativos", neg, f"{neg / total:.0%}", delta_color="inverse")
    c4.metric("Registros neutros", neu, f"{neu / total:.0%}", delta_color="off")


# Layout principal


def main() -> None:
    st.set_page_config(
        page_title="Dashboard Clínico — RPD",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Rastreio Emocional Automatizado (RPD)")
    st.caption(
        "Processamento de Linguagem Natural aplicado a diários pessoais · "
        "TCC — Engenharia de Software · UFC Campus Russas"
    )

    try:
        df = carregar_dados()
    except FileNotFoundError:
        st.error(
            f"Arquivo CSV não encontrado: `{_CSV_PATH}`. Execute o pipeline primeiro."
        )
        st.stop()

    personas = sorted(df["persona"].unique())

    # Uma aba por persona
    abas = st.tabs(personas)

    for aba, persona in zip(abas, personas):
        with aba:
            df_persona = preparar_persona(df, persona)

            # --- Cartões de resumo ---
            _metricas_resumo(df_persona)

            st.divider()

            # --- Linha 1: radar + mapa de calor ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Perfil de Somatização")
                st.caption(
                    "Frequência de sintomas físicos detectados pelo léxico clínico."
                )
                _grafico_radar_somatizacao(df_persona)

            with col2:
                st.subheader("Mapa de Calor: Oscilação de Humor")
                st.caption("Polaridade RoBERTa por dia e semana de monitoramento.")
                _grafico_mapa_calor(df_persona)

            st.divider()

            # --- Linha 2: evolução temporal (largura total) ---
            st.subheader("Evolução Longitudinal da Polaridade")
            st.caption(
                "Cada ponto representa um registro diário classificado pelo RoBERTa."
            )
            _grafico_evolucao_polaridade(df_persona)

            st.divider()

            # --- Linha 3: RPD + nuvem ---
            col3, col4 = st.columns([3, 2])
            with col3:
                st.subheader("Registro de Pensamentos Disfuncionais (RPD)")
                st.caption("Selecione o dia para inspecionar o registro estruturado.")
                _tabela_rpd(df_persona, persona)

            with col4:
                st.subheader("Nuvem de Contexto Clínico")
                st.caption("Substantivos, adjetivos e advérbios com maior frequência.")
                _nuvem_palavras(df_persona)


if __name__ == "__main__":
    main()
