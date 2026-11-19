"""
Integrante 3 - Agglomerative Clustering sobre fragmentos literarios.

Flujo:
    1. Probar las 13 recetas de representation.py en train (k=5, ward + euclidean).
    2. Con la mejor receta, probar combinaciones válidas de k, linkage y metric.
    3. Elegir la mejor configuración equilibrada en train (sin usar dev ni test).
    4. Guardar resumen por cluster (top 10 palabras y pasajes representativos),
       asignación aproximada por centroide más cercano en dev y test (una sola vez),
       y tabla externa cluster x book_id (solo para interpretación en el reporte).

Salida: TSV/Results/agglomerative/
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from representation import SEED, armar_vectorizador, cargar_textos, recetas_para
from results import (
    guardar_clusters,
    guardar_experimento,
    guardar_metrica,
    guardar_resumen_cluster,
)

ALGORITMO = "Agglomerative"

# Hiperparámetros de la guía (solo combinaciones válidas; ward solo acepta euclidean)
K_OPCIONES = [3, 4, 5, 6, 8, 10]
LINKAGE_METRIC = [
    ("ward", "euclidean"),
    ("average", "cosine"),
    ("complete", "cosine"),
]
MIN_TAMANO = 15  # umbral simple para descartar clusters diminutos


def espacio_nota(receta, metric):
    """Describe el espacio vectorial y la distancia usada."""
    espacio = f"SVD-{receta['svd']}+L2" if receta["svd"] else receta["rep"]
    return f"{espacio}, metric={metric}"


def vectorizar(receta, textos_train):
    """Ajusta el vectorizador solo con train y devuelve matriz densa para Agglomerative."""
    vec = armar_vectorizador(receta)
    if receta["svd"] is None:
        paso_rep = make_pipeline(vec, FunctionTransformer(lambda X: X.toarray()))
    else:
        paso_rep = vec
    X_train = paso_rep.fit_transform(textos_train)
    return paso_rep, X_train


def correr_experimento(exp_id, nombre_receta, receta, datos, k, linkage, metric):
    """Ajusta Agglomerative en train, evalúa silhouette y guarda el experimento."""
    paso_rep, X_train = vectorizar(receta, datos["train"]["texto"])
    modelo = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
    etiquetas = modelo.fit_predict(X_train)

    sil = float(silhouette_score(X_train, etiquetas, metric=metric))
    tamanos = [int((etiquetas == c).sum()) for c in range(k)]

    nota = f"tamanos={tamanos}"
    if min(tamanos) < MIN_TAMANO:
        nota += f" | cluster diminuto (min={min(tamanos)})"

    hp = {"k": k, "linkage": linkage, "metric": metric, "seed_svd": SEED}
    guardar_experimento(
        exp_id, ALGORITMO, nombre_receta, receta, hp,
        "silhouette (train)", round(sil, 6), seleccionado=False, notas=nota,
    )
    guardar_metrica(
        exp_id, ALGORITMO, "train", "silhouette", sil,
        nota=f"train; {espacio_nota(receta, metric)}",
    )
    return {
        "exp_id": exp_id,
        "nombre_receta": nombre_receta,
        "receta": receta,
        "k": k,
        "linkage": linkage,
        "metric": metric,
        "sil": sil,
        "tamanos": tamanos,
        "paso_rep": paso_rep,
        "X_train": X_train,
        "etiquetas": etiquetas,
    }


def asignar_centroide_cercano(X_train, etiquetas_train, X_nuevo, metric):
    """Regla aproximada para dev/test: asigna cada fragmento al centroide de train más cercano."""
    k = len(np.unique(etiquetas_train))
    centroides = np.vstack([X_train[etiquetas_train == c].mean(axis=0) for c in range(k)])
    distancias = pairwise_distances(X_nuevo, centroides, metric=metric)
    return distancias.argmin(axis=1)


def reportar_clusters(exp_id, res, df_train):
    """Calcula tamaño, top 10 palabras originales y pasajes representativos de cada cluster."""
    # En ambos casos (con o sin SVD), paso_rep[0] es el vectorizador original de palabras
    vec_palabras = res["paso_rep"][0]
    X_palabras = vec_palabras.transform(df_train["texto"])
    vocab = vec_palabras.get_feature_names_out()

    X_train = res["X_train"]
    etiquetas = res["etiquetas"]

    print(f"\nResumen de clusters ({exp_id}):")
    for c in range(res["k"]):
        mask = etiquetas == c
        tam = int(mask.sum())

        # Top 10 palabras promediando los vectores originales del cluster
        promedio = np.asarray(X_palabras[mask].mean(axis=0)).ravel()
        top_palabras = vocab[promedio.argsort()[::-1][:10]].tolist()

        # Pasajes más cercanos al centroide en el espacio del modelo
        centroide = X_train[mask].mean(axis=0, keepdims=True)
        dist = pairwise_distances(X_train[mask], centroide, metric=res["metric"]).ravel()
        pasajes = df_train.loc[mask, "fragment_id"].iloc[dist.argsort()[:3]].tolist()

        guardar_resumen_cluster(exp_id, ALGORITMO, c, tam, top_palabras, pasajes)
        print(f"  Cluster {c} (n={tam:2d}): {', '.join(top_palabras)} | pasajes={pasajes}")


def interpretacion_externa(etiquetas, df_split, nombre_split):
    """Tabla cluster x book_id y ARI/NMI solo para interpretación (nunca para elegir)."""
    ari = adjusted_rand_score(df_split["book_id"], etiquetas)
    nmi = normalized_mutual_info_score(df_split["book_id"], etiquetas)
    print(f"\nInterpretación externa en {nombre_split} (ARI={ari:.4f}, NMI={nmi:.4f}):")
    tabla = pd.crosstab(pd.Series(etiquetas, name="cluster"), df_split["book_id"])
    print(tabla.to_string())


def main():
    recetas = recetas_para("agglomerative")
    cache_datos = {}
    num_exp = 1

    # 1. Comparar las 13 recetas con configuración base (k=5, ward + euclidean)
    print("=== Paso 1: Comparación de 13 recetas en train (k=5, ward + euclidean) ===")
    resultados_recetas = []
    for nombre, receta in recetas.items():
        if receta["norm"] not in cache_datos:
            cache_datos[receta["norm"]] = cargar_textos(receta["norm"])
        exp_id = f"A{num_exp:02d}"
        num_exp += 1

        res = correr_experimento(exp_id, nombre, receta, cache_datos[receta["norm"]], 5, "ward", "euclidean")
        resultados_recetas.append(res)
        print(f"  {exp_id} {nombre:10s} sil={res['sil']:.4f} tamanos={res['tamanos']}")

    mejor_receta_res = max(resultados_recetas, key=lambda r: r["sil"])
    mejor_nombre = mejor_receta_res["nombre_receta"]
    mejor_receta = mejor_receta_res["receta"]
    datos = cache_datos[mejor_receta["norm"]]
    print(f"\nMejor receta: {mejor_nombre} ({mejor_receta_res['exp_id']}, sil={mejor_receta_res['sil']:.4f})")

    # 2. Búsqueda de hiperparámetros con la receta fija
    print(f"\n=== Paso 2: Búsqueda de hiperparámetros sobre {mejor_nombre} ===")
    resultados_hp = []
    for k in K_OPCIONES:
        for linkage, metric in LINKAGE_METRIC:
            exp_id = f"A{num_exp:02d}"
            num_exp += 1

            res = correr_experimento(exp_id, mejor_nombre, mejor_receta, datos, k, linkage, metric)
            resultados_hp.append(res)
            marca = " [diminuto]" if min(res["tamanos"]) < MIN_TAMANO else ""
            print(f"  {exp_id} k={k:2d} {linkage:8s}+{metric:9s} sil={res['sil']:.4f} tamanos={res['tamanos']}{marca}")

    # Elegir la mejor configuración sin clusters diminutos
    validos = [r for r in resultados_hp if min(r["tamanos"]) >= MIN_TAMANO]
    elegido = max(validos, key=lambda r: r["sil"])
    exp_final = elegido["exp_id"]

    # Marcar el experimento elegido como seleccionado=True en experiments.tsv
    hp_final = {"k": elegido["k"], "linkage": elegido["linkage"], "metric": elegido["metric"], "seed_svd": SEED}
    guardar_experimento(
        exp_final, ALGORITMO, mejor_nombre, mejor_receta, hp_final,
        "silhouette (train)", round(elegido["sil"], 6), seleccionado=True,
        notas=f"tamanos={elegido['tamanos']} | configuracion elegida",
    )
    print(f"\nConfiguración elegida: {exp_final} ({mejor_nombre}, {hp_final}, sil={elegido['sil']:.4f})")

    # 3. Guardar clusters de train y resumen interpretable
    guardar_clusters(
        exp_final, ALGORITMO, datos["train"]["fragment_id"],
        "train", elegido["etiquetas"], "fit_predict on train only",
    )
    reportar_clusters(exp_final, elegido, datos["train"])

    # 4. Asignación aproximada en dev y test (test se evalúa una sola vez al final)
    for split in ("dev", "test"):
        X_split = elegido["paso_rep"].transform(datos[split]["texto"])
        etiquetas_split = asignar_centroide_cercano(
            elegido["X_train"], elegido["etiquetas"], X_split, elegido["metric"],
        )
        guardar_clusters(
            exp_final, ALGORITMO, datos[split]["fragment_id"],
            split, etiquetas_split, "approximate assignment (nearest centroid)",
        )
        if len(np.unique(etiquetas_split)) > 1:
            sil_split = float(silhouette_score(X_split, etiquetas_split, metric=elegido["metric"]))
            guardar_metrica(
                exp_final, ALGORITMO, split, "silhouette", sil_split,
                nota=f"approximate assignment (nearest centroid); {espacio_nota(mejor_receta, elegido['metric'])}",
            )
            print(f"\nSilhouette aproximado en {split}: {sil_split:.4f}")
        interpretacion_externa(etiquetas_split, datos[split], split)

    interpretacion_externa(elegido["etiquetas"], datos["train"], "train")


if __name__ == "__main__":
    main()
