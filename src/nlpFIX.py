"""Normalizacion linguistica con spaCy (Fases 2 y 3)."""

import csv
import importlib
import re
from pathlib import Path

# Categorias gramaticales a conservar: sustantivos, verbos, adjetivos y adverbios
# Descarta articulos (DET), preposiciones (ADP), conjunciones (CCONJ/SCONJ) y pronombres (PRON)
KEPT_POS = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV"}

_nlp = None
_spacy = None

def get_spacy():
    """Carga y reutiliza el modelo de spaCy en espanol (preferencia: lg -> md -> sm)."""
    global _nlp, _spacy
    if _nlp is None:
        try:
            _spacy = importlib.import_module("spacy")
        except ModuleNotFoundError as exc:
            raise RuntimeError("spaCy no está instalado en el entorno de Python activo.") from exc
        for model in ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"]:
            try:
                _nlp = _spacy.load(model, disable=["ner"])
                print(f"[NLP] Modelo spaCy cargado exitosamente: {model}")
                break
            except OSError:
                continue
    return _nlp


def clean_text(text):
    """Reemplaza saltos de linea, tabulaciones y multiples espacios por uno solo."""
    text = re.sub(r"[\r\n\t]+", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()

def doc_to_lemmas(doc):
    """Filtra stopwords por POS y devuelve los lemas en minusculas."""
    lemmas = []
    for token in doc:
        if token.pos_ not in KEPT_POS or token.is_punct or token.is_space:
            continue
        # CAMBIO: .split() arregla "meterse" -> "meter él" (queda "meter")
        partes = token.lemma_.lower().split()
        if not partes:                      
            partes = token.text.lower().split()
        if partes:
            lemmas.append(partes[0])
    return lemmas

def tokenize_and_normalize(text):
    """Tokeniza, filtra stopwords por POS y extrae la lista de lemas en minusculas."""
    nlp = get_spacy()
    return doc_to_lemmas(nlp(clean_text(text)))  


def normalize_query(text):
    """Normaliza una consulta de usuario retornando una cadena de lemas (Fase 3)."""
    return " ".join(tokenize_and_normalize(text))


def normalize_corpus(
    raw_path="literature_raw_corpus.tsv",
    output_path="literature_normalized_corpusFIXED.tsv",
):
    """Normaliza el campo text del corpus usando nlp.pipe() y conserva el resto."""
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        records = list(reader)

    nlp = get_spacy()
    cleaned_texts = [clean_text(r["text"]) for r in records]

    print(f"[Fase 2] Normalizando {len(records)} fragmentos con spaCy...")
    for record, doc in zip(records, nlp.pipe(cleaned_texts, batch_size=100)):
        record["text"] = " ".join(doc_to_lemmas(doc)) 

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    print(f"[Fase 2] Corpus normalizado guardado en {output_path}")
    return records


if __name__ == "__main__":
    normalize_corpus()