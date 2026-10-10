"""Módulo de consolidación y gestión del corpus literario (Fase 1)."""

import csv
import glob
import os
from pathlib import Path
from typing import List, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None

OFFICIAL_COLUMNS = [
    "fragment_id",
    "book_id",
    "title",
    "author",
    "chapter/position",
    "text",
    "language",
    "source",
    "url",
    "keywords",
]


def build_raw_corpus(
    corpus_dir: Union[str, Path] = "corpus",
    output_path: Union[str, Path] = "raw_corpus.tsv",
) -> Union["pd.DataFrame", List[dict]]:
    """Busca recursivamente archivos .csv y .tsv en corpus_dir, consolida los

    fragmentos descartando columnas adicionales, reasigna fragment_id de forma
    secuencial de 1 a N, y exporta el corpus como archivo TSV (UTF-8).
    """
    corpus_path = Path(corpus_dir)
    pattern_csv = str(corpus_path / "**" / "*.csv")
    pattern_tsv = str(corpus_path / "**" / "*.tsv")

    raw_files = sorted(glob.glob(pattern_csv, recursive=True) + glob.glob(pattern_tsv, recursive=True))

    if not raw_files:
        raise FileNotFoundError(f"No se encontraron archivos .csv o .tsv en {corpus_dir}")

    consolidated_records: List[dict] = []

    for file_path in raw_files:
        delimiter = "\t" if file_path.endswith(".tsv") else ","
        with open(file_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                clean_row = {}
                for col in OFFICIAL_COLUMNS:
                    clean_row[col] = (row.get(col) or "").strip()
                consolidated_records.append(clean_row)

    # Reasignación de fragment_id de forma continua y global (1 a N)
    for idx, record in enumerate(consolidated_records, start=1):
        record["fragment_id"] = idx

    # Exportación a TSV (UTF-8)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFICIAL_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(consolidated_records)

    print(f"[Fase 1] Corpus consolidado exitosamente en: {output_path}")
    print(f"[Fase 1] Total de fragmentos unificados: {len(consolidated_records)}")

    if pd is not None:
        return pd.DataFrame(consolidated_records)
    return consolidated_records


if __name__ == "__main__":
    build_raw_corpus()
