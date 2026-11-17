"""
Paso 3 - Representación vectorial: recetas y utensilios comunes del equipo.

Este archivo NO corre experimentos. Da cuatro cosas que todos usan igual:
    1. RECETAS                     las 13 configuraciones (la base y una variación por factor)
    2. recetas_para(algoritmo)     las recetas válidas para cada algoritmo
    3. cargar_textos(norm)         textos normalizados de train, dev y test
    4. armar_vectorizador(receta)  el vectorizador listo para un Pipeline

Entrada : TSV/Normalized/normalized_corpus.tsv   (salida del Paso 2)
"""
from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

SEED = 3   # semilla común

BASE = Path(__file__).resolve().parent
CORPUS_NORMALIZADO = BASE / "TSV" / "Normalized" / "normalized_corpus.tsv"

# Piezas disponibles
REPRESENTACIONES = {
    "R1": "binaria",       # 1 si la palabra aparece, 0 si no
    "R2": "frecuencia",    # cuántas veces aparece
    "R3": "tfidf",         # frecuencia ponderada por rareza
}

# Vocabularios: umbrales elegidos con el análisis exploratorio (explore_vocab.py)
VOCABULARIOS = {
    "V1": dict(min_df=2, max_df=1.0),    # solo quita términos raros
    "V2": dict(min_df=2, max_df=0.85),   # quita raros y de relleno (ser, haber, no...)
    "V3": dict(min_df=5, max_df=0.85),   # vocabulario compacto (solo con unigramas)
}


# Recetario: base y cambiar SOLO un factor a la vez
_BASE = dict(norm="N3", rep="R3", ngram=(1, 1), vocab="V2", svd=None)


def _variar(**cambio):
    """Receta igual a la base, salvo el factor indicado."""
    return {**_BASE, **cambio}


RECETAS = {
    "base":     dict(_BASE),
    # Normalización
    "norm_N0":  _variar(norm="N0"),
    "norm_N1":  _variar(norm="N1"),
    "norm_N2":  _variar(norm="N2"),
    "norm_N4":  _variar(norm="N4"),
    # Representación
    "rep_R1":   _variar(rep="R1"),
    "rep_R2":   _variar(rep="R2"),
    # N-gramas
    "ngram_12": _variar(ngram=(1, 2)),
    "ngram_13": _variar(ngram=(1, 3)),
    # Vocabulario
    "vocab_V1": _variar(vocab="V1"),
    "vocab_V3": _variar(vocab="V3"),
    # SVD (solo clustering)
    "svd_100":  _variar(svd=100),
    "svd_200":  _variar(svd=200),
}

# Qué acepta cada algoritmo
#   LDA: solo conteos (R2) y sin SVD   |   NMF: solo TF-IDF (R3) y sin SVD
_REP_OBLIGATORIA = {"lda": "R2", "nmf": "R3"}


def recetas_para(algoritmo):
    """
    Devuelve las recetas válidas para un algoritmo:
        "kmeans", "agglomerative" -> las 13
        "lda"  -> 9 recetas, todas con R2 (conteos) y sin SVD
        "nmf"  -> 9 recetas, todas con R3 (TF-IDF) y sin SVD
    """
    algoritmo = algoritmo.lower()
    if algoritmo in ("kmeans", "agglomerative"):
        return dict(RECETAS)
    if algoritmo not in _REP_OBLIGATORIA:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    rep = _REP_OBLIGATORIA[algoritmo]
    validas = {}
    for nombre, receta in RECETAS.items():
        if nombre.startswith(("rep_", "svd_")):     # no aplican a LDA / NMF
            continue
        validas[nombre] = {**receta, "rep": rep}
    return validas



# Utils
def cargar_textos(norm):
    """
    Devuelve un diccionario {"train": df, "dev": df, "test": df}.
    Cada df tiene las columnas fragment_id, book_id y texto (en la normalización pedida).
    book_id es SOLO para interpretar resultados; nunca entra al modelo.
    """
    df = pd.read_csv(CORPUS_NORMALIZADO, sep="\t")
    if norm not in df.columns:
        raise ValueError(f"Normalización desconocida: {norm}")
    df = df.rename(columns={norm: "texto"})[["fragment_id", "book_id", "split", "texto"]]
    return {s: df[df["split"] == s].drop(columns="split").reset_index(drop=True)
            for s in ("train", "dev", "test")}


def armar_vectorizador(receta):
    """
    Arma la parte de representación del Pipeline según la receta:
        vectorizador (binaria / frecuencia / TF-IDF)  [+ SVD + normalización L2]
    Se ajusta con fit() SOLO en train; después se aplica con transform() a dev y test.
    """
    params = dict(
        tokenizer=str.split,     # los textos ya vienen tokenizados: solo cortar por espacio
        lowercase=False,         # la normalización ya decidió mayúsculas/minúsculas
        token_pattern=None,
        ngram_range=receta["ngram"],
        **VOCABULARIOS[receta["vocab"]],
    )
    rep = REPRESENTACIONES[receta["rep"]]
    if rep == "binaria":
        vec = CountVectorizer(binary=True, **params)
    elif rep == "frecuencia":
        vec = CountVectorizer(**params)
    else:
        vec = TfidfVectorizer(**params)

    if receta["svd"] is None:
        return vec

    # SVD y normalización L2: tras SVD los vectores pierden la longitud unitaria,
    # se vuelven a normalizar para que la distancia euclidiana equivalga al coseno.
    return make_pipeline(vec,
                         TruncatedSVD(n_components=receta["svd"], random_state=SEED),
                         Normalizer(copy=False))


def nombre_corto(receta):
    """Texto legible de una receta, útil para experiments.tsv. Ej.: 'N3|R3|1-1|V2|sinSVD'."""
    svd = f"SVD{receta['svd']}" if receta["svd"] else "sinSVD"
    ng = f"{receta['ngram'][0]}-{receta['ngram'][1]}"
    return f"{receta['norm']}|{receta['rep']}|{ng}|{receta['vocab']}|{svd}"


# Prueba rápida: python representation.py
if __name__ == "__main__":
    print(f"{'Receta':10} {'Configuración':26} {'train':>13} {'dev':>13} {'test':>13}")
    for nombre, receta in RECETAS.items():
        datos = cargar_textos(receta["norm"])
        vec = armar_vectorizador(receta)
        X_train = vec.fit_transform(datos["train"]["texto"])   # aprende solo con train
        X_dev = vec.transform(datos["dev"]["texto"])
        X_test = vec.transform(datos["test"]["texto"])
        forma = lambda X: f"{X.shape[0]}x{X.shape[1]}"
        print(f"{nombre:10} {nombre_corto(receta):26} "
              f"{forma(X_train):>13} {forma(X_dev):>13} {forma(X_test):>13}")

    print("\nRecetas por algoritmo:")
    for alg in ("kmeans", "agglomerative", "lda", "nmf"):
        print(f"  {alg:14} {len(recetas_para(alg))} recetas")