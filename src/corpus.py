"""Consolidacion del corpus literario (Fase 1)."""

import csv
import glob
from pathlib import Path

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


def build_raw_corpus(corpus_dir="corpus", output_path="raw_corpus.tsv"):
    # 1. Buscar todos los archivos csv y tsv recursivamente
    files = glob.glob(f"{corpus_dir}/**/*.csv", recursive=True) + glob.glob(
        f"{corpus_dir}/**/*.tsv", recursive=True
    )

    # 2. Leer cada archivo respetando el fragment_id original
    records = []
    for file_path in files:
        sep = "\t" if file_path.endswith(".tsv") else ","
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                record = {col: (row.get(col) or "").strip() for col in OFFICIAL_COLUMNS}
                record["fragment_id"] = int(record["fragment_id"])
                records.append(record)

    # 3. Ordenar por su fragment_id original (1 a 500)
    records.sort(key=lambda r: r["fragment_id"])

    # 4. Guardar archivo consolidado en formato TSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFICIAL_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    print(f"[Fase 1] {len(records)} fragmentos consolidados en {output_path}")
    return records


if __name__ == "__main__":
    build_raw_corpus()
