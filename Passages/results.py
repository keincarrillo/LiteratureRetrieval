"""
Paso 4 - Formato común de resultados del equipo.

Entrada : resultados de cada algoritmo, guardados con las funciones de este archivo
          en TSV/Results/<algoritmo>/   (kmeans, agglomerative, lda, nmf)
Salida  : TSV/Results/experiments.tsv, clustering_results.tsv, topic_results.tsv,
          metrics.tsv, cluster_summary.tsv y topic_words.tsv

Las funciones guardar_* se importan desde el script de cada algoritmo.
Ejecutar este archivo (python results.py) junta los resultados de los 4 algoritmos
en los entregables finales. 
"""
from pathlib import Path

import pandas as pd

from representation import nombre_corto

RESULTADOS = Path(__file__).resolve().parent / "TSV" / "Results"

# Nombre oficial, carpeta, tarea y prefijo de ID de cada algoritmo
ALGORITMOS = {
    "K-Means":       dict(carpeta="kmeans",        tarea="clustering", prefijo="K"),
    "Agglomerative": dict(carpeta="agglomerative", tarea="clustering", prefijo="A"),
    "LDA":           dict(carpeta="lda",           tarea="topics",     prefijo="L"),
    "NMF":           dict(carpeta="nmf",           tarea="topics",     prefijo="N"),
}

# Cómo se interpreta cada métrica
INTERPRETACION = {
    "silhouette": "higher is better",
    "perplexity": "lower is better",
    "reconstruction_error": "lower is better",
}

# Columnas de cada archivo (en el orden del enunciado)
COLUMNAS = {
    "experiments": ["experiment_id", "task", "method", "recipe", "normalization",
                    "representation", "ngram_range", "vocabulary", "dim_reduction",
                    "hyperparameters", "dev_metric", "dev_value", "selected", "notes"],
    "clustering_results": ["experiment_id", "algorithm", "fragment_id", "split",
                           "cluster_id", "assignment_note"],
    "topic_results": ["experiment_id", "algorithm", "fragment_id", "split",
                      "topic_id", "topic_weight"],
    "metrics": ["experiment_id", "algorithm", "split", "metric", "value",
                "interpretation", "note"],
    "cluster_summary": ["experiment_id", "algorithm", "cluster_id", "size",
                        "top_words", "representative_passages"],
    "topic_words": ["experiment_id", "algorithm", "topic_id", "top_words",
                    "representative_passages"],
}

# Columnas que identifican una fila: si se vuelve a guardar, la anterior se reemplaza
CLAVES = {
    "experiments": ["experiment_id"],
    "clustering_results": ["experiment_id", "split"],
    "topic_results": ["experiment_id", "split"],
    "metrics": ["experiment_id", "split", "metric"],
    "cluster_summary": ["experiment_id", "cluster_id"],
    "topic_words": ["experiment_id", "topic_id"],
}

REPRESENTACION_LEGIBLE = {"R1": "binary", "R2": "counts", "R3": "TF-IDF"}


# Funciones internas
def _validar(experiment_id, algoritmo):
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo '{algoritmo}' no válido. Usa uno de: {list(ALGORITMOS)}")
    prefijo = ALGORITMOS[algoritmo]["prefijo"]
    if not str(experiment_id).startswith(prefijo):
        raise ValueError(f"El ID '{experiment_id}' de {algoritmo} debe empezar con '{prefijo}' "
                         f"(ej. {prefijo}01)")


def _escribir(tabla, algoritmo, filas):
    """Agrega filas al archivo de la carpeta del algoritmo, reemplazando las repetidas."""
    nuevas = pd.DataFrame(filas, columns=COLUMNAS[tabla])
    carpeta = RESULTADOS / ALGORITMOS[algoritmo]["carpeta"]
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"{tabla}.tsv"

    if archivo.exists():
        viejas = pd.read_csv(archivo, sep="\t", dtype=str)
        clave = CLAVES[tabla]
        llaves_nuevas = set(map(tuple, nuevas[clave].astype(str).values))
        repetidas = viejas[clave].apply(lambda f: tuple(f.astype(str)) in llaves_nuevas, axis=1)
        nuevas = pd.concat([viejas[~repetidas], nuevas.astype(str)], ignore_index=True)

    nuevas.to_csv(archivo, sep="\t", index=False)


def _lista(valores):
    """Convierte una lista en texto: ['mar', 'barco'] -> 'mar, barco'."""
    return ", ".join(str(v) for v in valores)


def _hiperparametros(hp):
    """{'k': 5, 'n_init': 10} -> 'k=5; n_init=10'."""
    return "; ".join(f"{k}={v}" for k, v in hp.items()) if isinstance(hp, dict) else str(hp)


# Funciones para el equipo
def guardar_experimento(experiment_id, algoritmo, nombre_receta, receta, hiperparametros,
                        dev_metric, dev_value, seleccionado=False, notas=""):
    """
    Una fila de experiments.tsv.
        guardar_experimento("K01", "K-Means", "base", receta, {"k": 5, "n_init": 10},
                            "silhouette", 0.12, notas="clusters muy desbalanceados")
    Para Agglomerative, dev_metric puede ser "silhouette (train)".
    """
    _validar(experiment_id, algoritmo)
    ng = receta["ngram"]
    fila = [experiment_id, ALGORITMOS[algoritmo]["tarea"], algoritmo, nombre_receta,
            receta["norm"], REPRESENTACION_LEGIBLE[receta["rep"]], f"({ng[0]},{ng[1]})",
            receta["vocab"], f"SVD={receta['svd']}" if receta["svd"] else "none",
            _hiperparametros(hiperparametros), dev_metric, dev_value,
            "yes" if seleccionado else "no", notas or nombre_corto(receta)]
    _escribir("experiments", algoritmo, [fila])


def guardar_clusters(experiment_id, algoritmo, fragment_ids, split, cluster_ids, nota):
    """
    Cluster de cada fragmento de un split.
        K-Means:        nota="predict"
        Agglomerative:  nota="fit_predict on train only"  (o "approximate assignment" si
                        se implementa una regla para dev/test)
    """
    _validar(experiment_id, algoritmo)
    filas = [[experiment_id, algoritmo, f, split, int(c), nota]
             for f, c in zip(fragment_ids, cluster_ids)]
    _escribir("clustering_results", algoritmo, filas)


def guardar_temas(experiment_id, algoritmo, fragment_ids, split, pesos):
    """
    Peso de cada tema en cada fragmento (formato largo: una fila por fragmento y tema).
    `pesos` es la matriz que devuelve modelo.transform(textos): n_fragmentos x n_temas.
    """
    _validar(experiment_id, algoritmo)
    filas = [[experiment_id, algoritmo, f, split, t, round(float(w), 6)]
             for f, fila in zip(fragment_ids, pesos)
             for t, w in enumerate(fila)]
    _escribir("topic_results", algoritmo, filas)


def guardar_metrica(experiment_id, algoritmo, split, metrica, valor, nota=""):
    """
    metrica: "silhouette", "perplexity" o "reconstruction_error".
        guardar_metrica("L01", "LDA", "dev", "perplexity", 820.0, nota="fixed vocabulary")
    """
    _validar(experiment_id, algoritmo)
    fila = [experiment_id, algoritmo, split, metrica, round(float(valor), 6),
            INTERPRETACION.get(metrica, ""), nota]
    _escribir("metrics", algoritmo, [fila])


def guardar_resumen_cluster(experiment_id, algoritmo, cluster_id, tamano, top_palabras, pasajes):
    """Tamaño, top 10 palabras y fragment_id de pasajes representativos de UN cluster."""
    _validar(experiment_id, algoritmo)
    fila = [experiment_id, algoritmo, int(cluster_id), int(tamano),
            _lista(top_palabras[:10]), _lista(pasajes)]
    _escribir("cluster_summary", algoritmo, [fila])


def guardar_palabras_tema(experiment_id, algoritmo, topic_id, top_palabras, pasajes):
    """Top 10 palabras y fragment_id de pasajes representativos de UN tema."""
    _validar(experiment_id, algoritmo)
    fila = [experiment_id, algoritmo, int(topic_id), _lista(top_palabras[:10]), _lista(pasajes)]
    _escribir("topic_words", algoritmo, [fila])


# Juntar todo en los entregables finales: python results.py
def juntar():
    print(f"Juntando resultados de {RESULTADOS}\n")
    for tabla, columnas in COLUMNAS.items():
        partes = [pd.read_csv(RESULTADOS / a["carpeta"] / f"{tabla}.tsv", sep="\t", dtype=str)
                  for a in ALGORITMOS.values()
                  if (RESULTADOS / a["carpeta"] / f"{tabla}.tsv").exists()]
        if not partes:
            print(f"  {tabla + '.tsv':26} (sin datos todavía)")
            continue
        total = pd.concat(partes, ignore_index=True)[columnas]
        total.to_csv(RESULTADOS / f"{tabla}.tsv", sep="\t", index=False)
        por_alg = total.groupby(columnas[1] if tabla != "experiments" else "method").size()
        detalle = ", ".join(f"{alg}: {n}" for alg, n in por_alg.items())
        print(f"  {tabla + '.tsv':26} {len(total):6} filas  ({detalle})")


if __name__ == "__main__":
    juntar()