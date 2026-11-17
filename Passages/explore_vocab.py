"""
Análisis exploratorio del vocabulario (justificación de min_df, max_df y SVD).

Entrada : TSV/Normalized/normalized_corpus.tsv   (salida del Paso 2)
Salida  : TSV/Exploration/                        (tablas para el reporte)

Usa SOLO los fragmentos de train y no usa etiquetas (book_id, author...),
así que no hay fuga de información. Este análisis no elige la "mejor"
configuración: solo muestra qué valores tiene sentido probar.
La selección se hace después, con las métricas de dev.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SEED = 3          # semilla
CONFIG = "N3"     # normalización sobre la que se hace el análisis

# Los textos ya vienen tokenizados (separados por espacios): solo hay que cortar por espacio
TOKENS = dict(tokenizer=str.split, lowercase=False, token_pattern=None)

base = Path(__file__).resolve().parent
df = pd.read_csv(base / "TSV" / "Normalized" / "normalized_corpus.tsv", sep="\t")
train = df.loc[df["split"] == "train", CONFIG]
salida = base / "TSV" / "Exploration"
salida.mkdir(parents=True, exist_ok=True)

print(f"Análisis sobre {CONFIG}, {len(train)} fragmentos de train\n")

# 1. Tamaño del vocabulario según n-gramas y min_df
filas = []
for ng in [(1, 1), (1, 2), (1, 3)]:
    fila = {"ngram_range": str(ng)}
    for mn in [1, 2, 5]:
        vec = CountVectorizer(ngram_range=ng, min_df=mn, **TOKENS).fit(train)
        fila[f"min_df={mn}"] = len(vec.vocabulary_)
    filas.append(fila)
t1 = pd.DataFrame(filas)
t1.to_csv(salida / "vocab_min_df.tsv", sep="\t", index=False)
print("1) Tamaño del vocabulario (n-gramas x min_df)")
print(t1.to_string(index=False), "\n")

# 2. Términos que elimina cada max_df (con min_df=2, unigramas)
completo = set(CountVectorizer(min_df=2, **TOKENS).fit(train).vocabulary_)
filas = []
for mx in [1.0, 0.85, 0.7, 0.5]:
    vocab = set(CountVectorizer(min_df=2, max_df=mx, **TOKENS).fit(train).vocabulary_)
    quitados = sorted(completo - vocab)
    filas.append({"max_df": mx, "n_eliminados": len(quitados),
                  "terminos_eliminados": ", ".join(quitados)})
t2 = pd.DataFrame(filas)
t2.to_csv(salida / "vocab_max_df.tsv", sep="\t", index=False)
print("2) Términos eliminados por max_df (min_df=2, unigramas)")
for _, f in t2.iterrows():
    print(f"   max_df={f['max_df']}: {f['n_eliminados']} -> {f['terminos_eliminados']}")
print()

# 3. Varianza conservada por TruncatedSVD (TF-IDF, unigramas, min_df=2, max_df=0.85)
X = TfidfVectorizer(min_df=2, max_df=0.85, **TOKENS).fit_transform(train)
k_max = min(X.shape) - 1
svd = TruncatedSVD(n_components=min(300, k_max), random_state=SEED).fit(X)
acum = np.cumsum(svd.explained_variance_ratio_)
filas = [{"n_components": k, "varianza_conservada_%": round(acum[k - 1] * 100, 1)}
         for k in [50, 100, 200, 300] if k <= len(acum)]
t3 = pd.DataFrame(filas)
t3.to_csv(salida / "svd_varianza.tsv", sep="\t", index=False)
print(f"3) Varianza conservada por SVD (matriz {X.shape[0]} x {X.shape[1]}, "
      f"máximo n_components = {k_max})")
print(t3.to_string(index=False))
print(f"\nTablas guardadas en {salida}")