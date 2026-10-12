"""Representacion en Modelo de Espacio Vectorial (Fase 4). """

import argparse
import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

REPRESENTATIONS = ["frequency", "binary", "tfidf"]
FEATURES = ["unigram", "bigram"]


def build_vectorizer(representation: str = "tfidf", feature: str = "unigram"):
    """Crea el vectorizador de scikit-learn para una configuracion."""
    # 1. Rango de n-gramas
    if feature == "unigram":
        ngram_range = (1, 1)
    elif feature == "bigram":
        ngram_range = (2, 2)
    else:
        raise ValueError(f"Feature no valida: {feature}")

    # 2. Tipo de representacion
    # Se usa el token_pattern por defecto de sklearn: separa por letras/numeros
    # y descarta simbolos sueltos ("+", "=", "c.") y palabras de 1 letra.
    if representation == "binary":
        return CountVectorizer(binary=True, ngram_range=ngram_range)
    elif representation == "frequency":
        return CountVectorizer(binary=False, ngram_range=ngram_range)
    elif representation == "tfidf":
        return TfidfVectorizer(ngram_range=ngram_range)
    else:
        raise ValueError(f"Representacion no valida: {representation}")


def load_corpus(path):
    """Lee el corpus normalizado y devuelve (fragment_ids, textos)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [int(r["fragment_id"]) for r in rows], [r["text"] for r in rows]


def build_all(normalized_path="literature_normalized_corpus.tsv", out_dir="models"):
    """Entrena las 6 configuraciones y guarda cada una en un .pkl."""
    ids, texts = load_corpus(normalized_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[Fase 4] {len(texts)} documentos leidos de {normalized_path}")

    for representation in REPRESENTATIONS:
        for feature in FEATURES:
            name = f"{representation}_{feature}"
            vectorizer = build_vectorizer(representation, feature)
            matrix = vectorizer.fit_transform(texts)
            joblib.dump(
                {"vectorizer": vectorizer, "matrix": matrix, "fragment_ids": ids},
                Path(out_dir) / f"{name}.pkl",
            )
            print(f"  {name:18s} {matrix.shape[0]} docs x {matrix.shape[1]} terminos -> {name}.pkl")

    print(f"[Fase 4] 6 configuraciones guardadas en {out_dir}/")


def load_model(representation, feature, out_dir="models"):
    """Carga una configuracion guardada, p. ej. load_model('tfidf', 'unigram')."""
    return joblib.load(Path(out_dir) / f"{representation}_{feature}.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Construye las 6 representaciones VSM")
    p.add_argument("--input", default="literature_normalized_corpus.tsv")
    p.add_argument("--out", default="models")
    a = p.parse_args()

    build_all(a.input, a.out)