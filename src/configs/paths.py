"""Gerenciamento centralizado de caminhos físicos do projeto."""

from pathlib import Path


class PathConfig:
    """Caminhos físicos de arquivos e diretórios resolvidos a partir da raiz do projeto."""

    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    CORPUS_RAW_PATH: Path = RAW_DIR / "corpus_sintetico_tcc.csv"
    CORPUS_PROCESSED_PATH: Path = PROCESSED_DIR / "corpus_final_rpd_refinado.csv"
