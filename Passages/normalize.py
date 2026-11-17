"""
Paso 2 - Normalización del texto en 5 configuraciones (N0 a N4).

Entrada : TSV/documents.tsv                     (salida del Paso 1)
Salida  : TSV/Normalized/normalized_corpus.tsv  (fragment_id, book_id, split y una columna por configuración)

Configuraciones (cada una agrega un paso a la anterior):
    N0  Solo tokenizar (línea base)
    N1  N0 y limpieza: minúsculas, sin puntuación, números ni símbolos
    N2  N1 y quitar stopwords por categoría gramatical (filtro POS de la Etapa 1)
    N3  N2 y lematizar (equivale a la normalización de la Etapa 1)
    N4  N3 sin nombres propios (PROPN), para medir el peso de los personajes
"""
import re
from pathlib import Path

import pandas as pd
import spacy

MODELO_SPACY = "es_core_news_lg"   # modelo fijo

# Categorías gramaticales que se CONSERVAN en N2 (igual que en la Etapa 1):
# sustantivos, nombres propios, verbos, auxiliares, adjetivos y adverbios.
# Se descartan artículos, preposiciones, conjunciones, pronombres, etc.
KEPT_POS = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV"}

CONFIGS = ["N0", "N1", "N2", "N3", "N4"]


def limpiar_espacios(texto):
    """Reemplaza saltos de línea, tabuladores y espacios repetidos por un solo espacio."""
    texto = re.sub(r"[\r\n\t]+", " ", texto or "")
    return re.sub(r"\s+", " ", texto).strip()


def es_palabra(token):
    """True si el token es una palabra: tiene al menos una letra y no es número."""
    return any(c.isalpha() for c in token.text) and not token.like_num


def lema(token):
    """Lema en minúsculas. Si spaCy devuelve dos palabras ("meter él"), se queda con la primera."""
    partes = token.lemma_.lower().split() or token.text.lower().split()
    return partes[0]


def normalizar(doc):
    """Devuelve un diccionario con las 5 versiones del fragmento."""
    n0, n1, n2, n3, n4 = [], [], [], [], []

    for tok in doc:
        if tok.is_space:
            continue

        # N0: solo tokenizar (se conservan mayúsculas, puntuación y números)
        n0.append(tok.text)

        # N1: limpieza
        if not es_palabra(tok):
            continue
        n1.append(tok.text.lower())

        # N2: quitar stopwords por categoría gramatical
        if tok.pos_ not in KEPT_POS:
            continue
        n2.append(tok.text.lower())

        # N3: lematizar
        n3.append(lema(tok))

        # N4: sin nombres propios
        if tok.pos_ != "PROPN":
            n4.append(lema(tok))

    return {c: " ".join(t) for c, t in zip(CONFIGS, [n0, n1, n2, n3, n4])}


def main():
    base = Path(__file__).resolve().parent
    entrada = base / "TSV" / "documents.tsv"
    salida = base / "TSV" / "Normalized" / "normalized_corpus.tsv"
    salida.parent.mkdir(parents=True, exist_ok=True)   # crea TSV/Normalized/ si no existe

    df = pd.read_csv(entrada, sep="\t")

    # Se desactiva NER: no se necesita (los nombres propios se detectan con POS = PROPN)
    nlp = spacy.load(MODELO_SPACY, disable=["ner"])
    print(f"[Paso 2] spaCy {spacy.__version__} | modelo {MODELO_SPACY} "
          f"{nlp.meta['version']}")
    print(f"[Paso 2] Normalizando {len(df)} fragmentos...")

    textos = [limpiar_espacios(t) for t in df["text"]]
    filas = [normalizar(doc) for doc in nlp.pipe(textos, batch_size=50)]

    norm = pd.DataFrame(filas)
    resultado = pd.concat([df[["fragment_id", "book_id", "split"]], norm], axis=1)

    # Comprobaciones
    assert len(resultado) == len(df)
    for c in CONFIGS:
        vacios = (resultado[c].str.len() == 0).sum()
        assert vacios == 0, f"{c} tiene {vacios} fragmentos vacíos"

    resultado.to_csv(salida, sep="\t", index=False)
    print(f"[Paso 2] Guardado en {salida}\n")

    # Resumen: tokens promedio y tamaño de vocabulario (en train) por configuración
    train = resultado[resultado["split"] == "train"]
    print(f"{'Config':6} {'Tokens/fragmento':>17} {'Vocabulario (train)':>20}")
    for c in CONFIGS:
        tokens = resultado[c].str.split()
        vocab = set(w for fila in train[c].str.split() for w in fila)
        print(f"{c:6} {tokens.str.len().mean():17.1f} {len(vocab):20,d}")

    # Ejemplo de un fragmento
    ej = resultado.iloc[0]
    print(f"\nEjemplo (fragment_id {ej['fragment_id']}, primeras 12 palabras):")
    for c in CONFIGS:
        print(f"  {c}: {' '.join(ej[c].split()[:12])}")


if __name__ == "__main__":
    main()