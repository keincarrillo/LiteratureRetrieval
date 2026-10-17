"""Fase 5: recuperación de fragmentos mediante VSM y similitud coseno."""

import argparse
import csv
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .nlp import normalize_query
from .vsm import FEATURES, REPRESENTATIONS, load_model


ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "literature_raw_corpus.tsv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TOP_K = 10
PREVIEW_LENGTH = 350


def load_raw_documents():
    """Lee el corpus original e indexa los fragmentos por ID."""
    if not CORPUS.is_file():
        raise FileNotFoundError(f"No se encontró el corpus: {CORPUS}")

    documents = {}
    with CORPUS.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"fragment_id", "text", "title", "author"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en el corpus: {sorted(missing)}")

        for row in reader:
            fragment_id = row["fragment_id"].strip()
            if not fragment_id or fragment_id in documents:
                raise ValueError(f"ID vacío o duplicado: {fragment_id!r}")
            documents[fragment_id] = row

    return documents


def retrieve(query, representation="tfidf", feature="unigram"):
    """Devuelve la consulta normalizada y los diez fragmentos mejor puntuados."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("La consulta no puede estar vacía")
    if representation not in REPRESENTATIONS or feature not in FEATURES:
        raise ValueError("Representación o tipo de n-grama no válido")

    normalized_query = normalize_query(query)
    if not normalized_query.strip():
        return normalized_query, []

    model_path = MODELS / f"{representation}_{feature}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    model = load_model(representation, feature, out_dir=str(MODELS))
    matrix = model["matrix"]
    fragment_ids = model["fragment_ids"]
    if matrix.shape[0] != len(fragment_ids):
        raise ValueError("La matriz y los IDs del modelo no coinciden")

    query_vector = model["vectorizer"].transform([normalized_query])
    if query_vector.nnz == 0:
        return normalized_query, []

    scores = cosine_similarity(query_vector, matrix).ravel()
    indices = np.argsort(-scores, kind="stable")[:TOP_K]
    documents = load_raw_documents()
    results = []

    for rank, index in enumerate(indices, 1):
        fragment_id = str(fragment_ids[index]).strip()
        if fragment_id not in documents:
            raise ValueError(f"El fragmento {fragment_id} no existe en el corpus original")

        row = documents[fragment_id]
        results.append({
            "rank": rank,
            "fragment_id": fragment_id,
            "score": float(scores[index]),
            "book_id": row.get("book_id", ""),
            "title": row["title"],
            "author": row["author"],
            "position": row.get("chapter/position", ""),
            "keywords": (row.get("keywords") or "").strip(),
            "text": row["text"],
            "language": row.get("language", ""),
            "source": row.get("source", ""),
            "url": row.get("url", ""),
        })

    return normalized_query, results


def print_results(query, normalized_query, results, representation, feature):
    """Muestra en consola el ranking y una vista previa de cada fragmento."""
    print("\n========== RECUPERACIÓN DE INFORMACIÓN (FASE 5) ==========")
    print(f"Consulta original   : {query}")
    print(f"Consulta normalizada: {normalized_query}")
    print(f"Modelo              : {representation} / {feature}")
    print(f"Resultados          : {len(results)}\n")

    if not results:
        print("No se encontraron coincidencias en el vocabulario del modelo.")

    for result in results:
        text = re.sub(r"\s+", " ", result["text"].strip())
        if len(text) > PREVIEW_LENGTH:
            text = text[:PREVIEW_LENGTH].rsplit(" ", 1)[0] + "..."

        print(f"[{result['rank']}] Fragmento ID: {result['fragment_id']}")
        print(f"    Similitud coseno: {result['score']:.4f}")
        print(f"    Obra            : {result['title']}")
        print(f"    Autor           : {result['author']}")
        print(f"    Posición        : {result['position']}")
        print(f"    Palabras clave  : {result['keywords'] or 'No disponibles'}")
        print(f"    Texto           : {text}")
        print("-" * 70)


def save_results_txt(query, normalized_query, results, representation, feature):
    """Guarda automáticamente los fragmentos completos en un TXT único."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    slug = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")[:45].rstrip("_") or "consulta"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = RESULTS / f"{timestamp}_{slug}_{representation}_{feature}.txt"

    lines = [
        "RESULTADOS DE RECUPERACIÓN DE INFORMACIÓN - FASE 5",
        "=" * 72,
        f"Consulta original    : {query}",
        f"Consulta normalizada : {normalized_query}",
        f"Representación VSM   : {representation}",
        f"Características      : {feature}",
        f"Resultados obtenidos : {len(results)}",
        "=" * 72,
        "",
    ]

    if not results:
        lines.append("No se encontraron resultados para esta consulta.")

    for result in results:
        lines.extend([
            f"RESULTADO {result['rank']}",
            "-" * 72,
            f"Fragmento ID     : {result['fragment_id']}",
            f"Similitud coseno : {result['score']:.6f}",
            f"Obra             : {result['title']}",
            f"Autor            : {result['author']}",
            f"Posición         : {result['position']}",
            f"Palabras clave   : {result['keywords'] or 'No disponibles'}",
            f"Book ID          : {result['book_id']}",
            f"Idioma           : {result['language']}",
            f"Fuente           : {result['source']}",
            f"URL              : {result['url']}",
            "",
            "TEXTO COMPLETO:",
            result["text"],
            "",
            "=" * 72,
            "",
        ])

    with path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Recupera los 10 fragmentos más relevantes")
    parser.add_argument("--query", "-q", required=True, help="Consulta en lenguaje natural")
    parser.add_argument("--representation", "-r", choices=REPRESENTATIONS, default="tfidf")
    parser.add_argument("--feature", "-f", choices=FEATURES, default="unigram")
    args = parser.parse_args()

    try:
        normalized_query, results = retrieve(args.query, args.representation, args.feature)
        print_results(args.query, normalized_query, results, args.representation, args.feature)
        path = save_results_txt(args.query, normalized_query, results, args.representation, args.feature)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.error(str(error))

    print(f"\nResultados completos guardados en: {path}")


if __name__ == "__main__":
    main()
