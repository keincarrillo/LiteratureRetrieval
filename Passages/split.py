"""
Paso 1 - Partición train / dev / test por capítulo, dentro de cada libro.

Entrada : ../LiteratureRetrieval/literature_raw_corpus.tsv  (corpus de la Etapa 1)
Salida  : TSV/documents.tsv  (mismo corpus; se añaden las columnas chapter_group y split)
"""
import random
import pandas as pd
from pathlib import Path

SEED = 3            # semilla común 
P_TEST = 0.15       # proporción objetivo de test
P_DEV = 0.15        # proporción objetivo de dev (el resto va a train)

# 1. Cargar el corpus
BASE = Path(__file__).resolve().parent          # carpeta Passages/
df = pd.read_csv(BASE.parent / "literature_raw_corpus.tsv", sep="\t")

# 2. Limpiar el capítulo: en 1984 viene "Parte Uno, capítulo I; páginas 5–6",
#    así que nos quedamos con lo que está antes del ';'
capitulo = df["chapter/position"].str.split(";").str[0].str.strip()

# 3. El grupo es libro y capítulo (para no mezclar capítulos con el mismo nombre en libros distintos)
df["chapter_group"] = df["book_id"] + " | " + capitulo

# 4. Repartir capítulos completos, libro por libro
rng = random.Random(SEED)
split_de_grupo = {}

for book_id, frag_libro in df.groupby("book_id"):
    # capítulos del libro y cuántos fragmentos tiene cada uno
    tam = frag_libro.groupby("chapter_group").size()
    capitulos = sorted(tam.index)          # orden fijo antes de barajar
    rng.shuffle(capitulos)                 # barajar con la semilla

    meta_test = round(len(frag_libro) * P_TEST)   # 15 de 100
    meta_dev = round(len(frag_libro) * P_DEV)     # 15 de 100
    n_test = n_dev = 0

    # Regla "solo si cabe": un capítulo entra a test o dev únicamente
    # si no hace que ese montón pase de su meta; si no cabe en ninguno, va a train.
    for cap in capitulos:
        if n_test + tam[cap] <= meta_test:
            split_de_grupo[cap] = "test"
            n_test += tam[cap]
        elif n_dev + tam[cap] <= meta_dev:
            split_de_grupo[cap] = "dev"
            n_dev += tam[cap]
        else:
            split_de_grupo[cap] = "train"

df["split"] = df["chapter_group"].map(split_de_grupo)

# 5. Comprobaciones
assert df["split"].notna().all()                                  # todos tienen montón
assert df.groupby("chapter_group")["split"].nunique().max() == 1  # ningún capítulo partido
assert df["fragment_id"].is_unique                                # ids intactos

# 6. Guardar: primero las columnas obligatorias (sección 11 del enunciado),
#    después los metadatos extra de la Etapa 1
obligatorias = ["fragment_id", "book_id", "title", "author", "text", "split"]
extra = [c for c in df.columns if c not in obligatorias]
df = df[obligatorias + extra]
(BASE / "TSV").mkdir(exist_ok=True)
df.to_csv(BASE / "TSV" / "documents.tsv", sep="\t", index=False)

# 7. Resumen
print(pd.crosstab(df["book_id"], df["split"], margins=True, margins_name="Total")
        [["train", "dev", "test", "Total"]])
print()
print((df["split"].value_counts(normalize=True) * 100).round(1).rename("%"))