import argparse
import csv
import json
import os
import re
import sys

ARCHIVO_CSV = "corpus_kundera.csv"
ARCHIVO_SALIDA = "corpus_kundera.csv"
BOOK_ID_CFG = None

# Diccionario amplio de motivos y conceptos de la obra
MOTIVOS_KUNDERA = {
    # Personajes
    "tomas": r"\btom[aá]s\b",
    "teresa": r"\bteresa\b",
    "sabina": r"\bsabina\b",
    "franz": r"\bfranz\b",
    "karenin": r"\bkarenin\b",
    "simon": r"\b(sim[oó]n|hijo)\b",
    "marie_claude": r"\bmarie-claude\b",
    # Conceptos filosóficos
    "eterno_retorno": r"\beterno retorno\b",
    "levedad": r"\blevedad\b",
    "peso": r"\bpeso\b",
    "kitsch": r"\bkitsch\b",
    "vertigo": r"\bv[eé]rtigo\b",
    "compasion": r"\bcompasi[oó]n\b",
    "azar": r"\b(azar|casualidad|coincidencia)\b",
    "metafora": r"\bmet[aá]fora\b",
    "destino": r"\bdestino\b",
    # Símbolos, cuerpo e intimidad
    "sombrero_hongo": r"\bsombrero(\s+hongo)?\b",
    "espejo": r"\bespejo\b",
    "cuerpo": r"\bcuerpo\b",
    "alma": r"\balma\b",
    "desnudez": r"\bdesnud[oa]s?\b",
    "celos": r"\bcelos\b",
    "infidelidad": r"\b(infidelidad|amante[s]?|adulterio)\b",
    "erotismo": r"\berotismo\b",
    "pesadillas": r"\b(sue[ñn]o[s]?|pesadilla[s]?)\b",
    # Política y sociedad
    "praga": r"\bpraga\b",
    "ocupacion_rusa": r"\b(tanques?|ruso[s]?|ocupaci[oó]n|invasi[oó]n)\b",
    "disidencia_politica": (
        r"\b(polic[ií]a|interrogatori|firma|petici[oó]n|r[eé]gimen)\b"
    ),
    "limpiacristales": r"\b(limpiacristales|ventanas?|cirujano|m[eé]dico)\b",
    # Lugares y entorno
    "zurich": r"\bz[uú]rich\b",
    "ginebra": r"\bginebra\b",
    "campo_pueblo": r"\b(campo|pueblo|cooperativa|tractor)\b",
    # Existencia, religión y muerte
    "religion_fe": r"\b(dios|iglesia|fe|creyente[s]?|religi[oó]n)\b",
    "musica": r"\bm[uú]sica\b",
    "muerte": r"\b(muerte|morir|agon[ií]a|entierro)\b",
}

COLUMNAS_OFICIALES = [
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


def extraer_todas_las_keywords(texto):
    t_min = texto.lower()
    encontradas = []

    for tag, patron in MOTIVOS_KUNDERA.items():
        if re.search(patron, t_min):
            encontradas.append(tag)

    if not encontradas:
        return "general"

    return ", ".join(encontradas)


def cargar_config(ruta_config=None):
    global ARCHIVO_CSV, ARCHIVO_SALIDA, MOTIVOS_KUNDERA, BOOK_ID_CFG
    if ruta_config is None:
        return
    try:
        with open(ruta_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[WARN] No se pudo cargar config {ruta_config}: {e}")
        return

    base_dir = os.path.dirname(os.path.abspath(ruta_config))
    ARCHIVO_CSV = cfg.get("csv") or ARCHIVO_CSV
    ARCHIVO_SALIDA = cfg.get("csv") or ARCHIVO_CSV
    motivos = cfg.get("motivos_keywords") or cfg.get("motivos") or {}
    if motivos:
        if isinstance(motivos, list):
            MOTIVOS_KUNDERA = {f"kw_{i}": m for i, m in enumerate(motivos)}
        else:
            MOTIVOS_KUNDERA = dict(motivos)
    BOOK_ID_CFG = cfg.get("book_id") or BOOK_ID_CFG

    if not os.path.isabs(ARCHIVO_CSV):
        ARCHIVO_CSV = os.path.join(base_dir, ARCHIVO_CSV)
    if not os.path.isabs(ARCHIVO_SALIDA):
        ARCHIVO_SALIDA = os.path.join(base_dir, ARCHIVO_SALIDA)


def actualizar_corpus():
    if not os.path.exists(ARCHIVO_CSV):
        print(f"Error: No existe el archivo '{ARCHIVO_CSV}'.")
        sys.exit(1)

    with open(ARCHIVO_CSV, "r", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    if not filas:
        print("El archivo CSV no contiene registros.")
        sys.exit(1)

    for fila in filas:
        if not fila.get("book_id"):
            if BOOK_ID_CFG:
                fila["book_id"] = BOOK_ID_CFG
        fila["keywords"] = extraer_todas_las_keywords(fila.get("text", ""))

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(
            f, fieldnames=COLUMNAS_OFICIALES, quoting=csv.QUOTE_MINIMAL
        )
        escritor.writeheader()
        escritor.writerows(filas)

    print(
        f"Se actualizaron todas las keywords de {len(filas)} fragmentos en '{ARCHIVO_SALIDA}'.\n"
    )
    print("Muestra de los primeros registros:")
    for f in filas[:5]:
        print(
            f"ID {f['fragment_id']} ({f['chapter/position']}):\n  -> {f['keywords']}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalcula keywords para corpus CSV")
    parser.add_argument("--config", "-c", type=str, help="Ruta a book.json")
    parser.add_argument("--csv", type=str, help="Ruta al CSV (anula config)")
    parser.add_argument("--out", type=str, help="Ruta de salida (anula config)")
    args = parser.parse_args()

    cargar_config(args.config)
    if args.csv:
        ARCHIVO_CSV = os.path.abspath(args.csv)
    if args.out:
        ARCHIVO_SALIDA = os.path.abspath(args.out)

    actualizar_corpus()
