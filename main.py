import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from src.corpus import build_raw_corpus
from src.nlp import normalize_corpus, normalize_query, tokenize_and_normalize


def run_phase1():
    print("FASE 1: Corpus crudo completo")
    df = build_raw_corpus()
    return df


def run_phase2():
    print("FASE 2: Normalizacion con spaCy")
    df = normalize_corpus()
    return df


def test_query_normalization(query: str):
    print("FASE 3: Normalizacion de Consulta Libre")
    print(f"Consulta original : {query}")
    tokens = tokenize_and_normalize(query)
    print(f"Tokens filtrados y lematizados: {tokens}")
    normalized = " ".join(tokens)
    print(f"Texto procesado   : {normalized}")
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="LiteratureRetrieval CLI - Procesamiento de Lenguaje Natural"
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Ejecutar Fase 1: Consolidar archivos de corpus en raw_corpus.tsv",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Ejecutar Fase 2: Normalizar el corpus linguisticamente en normalized_corpus.tsv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecutar Fase 1 y Fase 2 de manera consecutiva",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Probar normalizacion de una consulta libre de texto (ej. 'descripcion de una tormenta en el mar')",
    )

    args = parser.parse_args()

    if not (args.phase1 or args.phase2 or args.all or args.query):
        args.all = True

    if args.phase1 or args.all:
        run_phase1()

    if args.phase2 or args.all:
        run_phase2()

    if args.query:
        test_query_normalization(args.query)


if __name__ == "__main__":
    main()
