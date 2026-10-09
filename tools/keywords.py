import csv
import os
import re
import sys

ARCHIVO_CSV = "corpus_kundera.csv"
ARCHIVO_SALIDA = "corpus_kundera.csv"

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
        fila["book_id"] = "kundera_levedad"
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
    actualizar_corpus()
