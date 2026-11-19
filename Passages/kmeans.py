"""
Integrante 2 - K-Means sobre fragmentos literarios.

Flujo:
    1. Probar las 13 recetas de representation.py en train con hiperparámetros por defecto.
    2. Con la mejor receta (elegida por silhouette en dev), buscar n_clusters, init,
       n_init y max_iter. La selección usa SOLO dev; test se evalúa una sola vez al final.
    3. Guardar resumen por cluster (top 10 palabras y pasajes representativos), la
       asignación de train/dev/test hecha con predict() y la tabla externa cluster x book_id
       únicamente para interpretar el reporte.

Salida: TSV/Results/kmeans/
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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

ALGORITMO = "K-Means"

# Hiperparámetros por defecto para la comparación de recetas
K_RECETAS = 5
INIT_DEFECTO = "k-means++"
N_INIT_DEFECTO = 10
MAX_ITER_DEFECTO = 300

# Grid de hiperparámetros (la semilla se mantiene fija en SEED)
K_OPCIONES = [3, 4, 5, 6, 8, 10]
INIT_OPCIONES = ["k-means++", "random"]
N_INIT_OPCIONES = [10, 20]
MAX_ITER_OPCIONES = [100, 300]

MIN_TAMANO = 15  # mismo criterio que Agglomerative para descartar clusters diminutos


def espacio_nota(receta):
    """Describe el espacio vectorial usado por la métrica."""
    return f"SVD-{receta['svd']}+L2" if receta["svd"] else receta["rep"]


def vectorizar(receta, textos_train):
    """Ajusta la representación solo con train y devuelve matrices densas."""
    vec = armar_vectorizador(receta)
    if receta["svd"] is None:
        paso_rep = make_pipeline(vec, FunctionTransformer(lambda X: X.toarray()))
    else:
        # armar_vectorizador ya incluye TruncatedSVD + Normalizer (L2)
        paso_rep = vec
    X_train = paso_rep.fit_transform(textos_train)
    return paso_rep, X_train


def obtener_representacion(nombre_receta, receta, datos, cache):
    """Cachea la representación de una receta para reutilizarla en todo el grid."""
    if nombre_receta not in cache:
        paso_rep, X_train = vectorizar(receta, datos["train"]["texto"])
        cache[nombre_receta] = {
            "paso_rep": paso_rep,
            "X_train": X_train,
            "X_dev": paso_rep.transform(datos["dev"]["texto"]),
            "X_test": paso_rep.transform(datos["test"]["texto"]),
        }
    return cache[nombre_receta]


def silhouette(X, etiquetas):
    """Silhouette euclidiana; NaN si la partición tiene menos de dos clusters."""
    if len(np.unique(etiquetas)) < 2:
        return float("nan")
    return float(silhouette_score(X, etiquetas))


def correr_experimento(exp_id, nombre_receta, receta, datos, cache, k, init, n_init, max_iter):
    """Ajusta K-Means en train, evalúa en dev con predict() y registra el experimento."""
    rep = obtener_representacion(nombre_receta, receta, datos, cache)
    X_train = rep["X_train"]
    X_dev = rep["X_dev"]

    modelo = KMeans(
        n_clusters=k,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=SEED,
    )
    etiquetas_train = modelo.fit_predict(X_train)
    etiquetas_dev = modelo.predict(X_dev)

    sil_dev = silhouette(X_dev, etiquetas_dev)
    tamanos = [int((etiquetas_train == c).sum()) for c in range(k)]

    nota = f"tamanos={tamanos}"
    if min(tamanos) < MIN_TAMANO:
        nota += f" | cluster diminuto (min={min(tamanos)})"
    if not np.isfinite(sil_dev):
        nota += " | dev con un solo cluster predicho"

    hp = {"k": k, "init": init, "n_init": n_init, "max_iter": max_iter, "seed": SEED}
    valor_dev = round(sil_dev, 6) if np.isfinite(sil_dev) else "nan"
    guardar_experimento(
        exp_id, ALGORITMO, nombre_receta, receta, hp,
        "silhouette (dev)", valor_dev, seleccionado=False, notas=nota,
    )
    if np.isfinite(sil_dev):
        guardar_metrica(
            exp_id, ALGORITMO, "dev", "silhouette", sil_dev,
            nota=f"predict en dev; {espacio_nota(receta)}",
        )

    return {
        "exp_id": exp_id,
        "nombre_receta": nombre_receta,
        "receta": receta,
        "k": k,
        "init": init,
        "n_init": n_init,
        "max_iter": max_iter,
        "sil": sil_dev,
        "tamanos": tamanos,
        "modelo": modelo,
        "paso_rep": rep["paso_rep"],
        "X_train": X_train,
        "etiquetas": etiquetas_train,
    }


def reportar_clusters(exp_id, res, df_train):
    """Calcula tamaño, top 10 palabras originales y pasajes representativos de cada cluster."""
    # Con y sin SVD, paso_rep[0] es el vectorizador original de palabras
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
        dist = pairwise_distances(X_train[mask], centroide).ravel()
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
    recetas = recetas_para("kmeans")
    cache = {}
    datos_por_norm = {}
    num_exp = 1

    def datos(norm):
        if norm not in datos_por_norm:
            datos_por_norm[norm] = cargar_textos(norm)
        return datos_por_norm[norm]

    # 1. Comparar las 13 recetas con hiperparámetros por defecto
    print("=== Paso 1: Comparación de 13 recetas en train, selección por silhouette en dev ===")
    resultados_recetas = []
    for nombre, receta in recetas.items():
        exp_id = f"K{num_exp:02d}"
        num_exp += 1
        res = correr_experimento(
            exp_id, nombre, receta, datos(receta["norm"]), cache,
            K_RECETAS, INIT_DEFECTO, N_INIT_DEFECTO, MAX_ITER_DEFECTO,
        )
        resultados_recetas.append(res)
        print(f"  {exp_id} {nombre:10s} sil(dev)={res['sil']:.4f} tamanos={res['tamanos']}")

    ganadores = [r for r in resultados_recetas if np.isfinite(r["sil"])]
    mejor_receta_res = max(ganadores, key=lambda r: r["sil"])
    mejor_nombre = mejor_receta_res["nombre_receta"]
    mejor_receta = mejor_receta_res["receta"]
    datos_elegidos = datos(mejor_receta["norm"])
    print(f"\nMejor receta: {mejor_nombre} "
          f"({mejor_receta_res['exp_id']}, sil(dev)={mejor_receta_res['sil']:.4f})")

    # 2. Búsqueda de hiperparámetros con la receta fija
    print(f"\n=== Paso 2: Búsqueda de hiperparámetros sobre {mejor_nombre} (silhouette en dev) ===")
    resultados_hp = []
    for k in K_OPCIONES:
        for init in INIT_OPCIONES:
            for n_init in N_INIT_OPCIONES:
                for max_iter in MAX_ITER_OPCIONES:
                    exp_id = f"K{num_exp:02d}"
                    num_exp += 1
                    res = correr_experimento(
                        exp_id, mejor_nombre, mejor_receta, datos_elegidos, cache,
                        k, init, n_init, max_iter,
                    )
                    resultados_hp.append(res)
                    marca = " [diminuto]" if min(res["tamanos"]) < MIN_TAMANO else ""
                    print(f"  {exp_id} k={k:2d} {init:9s} n_init={n_init:2d} "
                          f"max_iter={max_iter:3d} sil(dev)={res['sil']:.4f} "
                          f"tamanos={res['tamanos']}{marca}")

    validos = [r for r in resultados_hp
               if np.isfinite(r["sil"]) and min(r["tamanos"]) >= MIN_TAMANO]
    elegido = max(validos, key=lambda r: r["sil"])
    exp_final = elegido["exp_id"]

    hp_final = {
        "k": elegido["k"], "init": elegido["init"], "n_init": elegido["n_init"],
        "max_iter": elegido["max_iter"], "seed": SEED,
    }
    guardar_experimento(
        exp_final, ALGORITMO, mejor_nombre, mejor_receta, hp_final,
        "silhouette (dev)", round(elegido["sil"], 6), seleccionado=True,
        notas=f"tamanos={elegido['tamanos']} | configuracion elegida",
    )
    print(f"\nConfiguración elegida: {exp_final} ({mejor_nombre}, {hp_final}, "
          f"sil(dev)={elegido['sil']:.4f})")

    # 3. Asignaciones nativas con predict() y resumen interpretable
    guardar_clusters(
        exp_final, ALGORITMO, datos_elegidos["train"]["fragment_id"],
        "train", elegido["etiquetas"], "predict",
    )
    reportar_clusters(exp_final, elegido, datos_elegidos["train"])

    for split in ("dev", "test"):
        X_split = cache[mejor_nombre][f"X_{split}"]
        etiquetas_split = elegido["modelo"].predict(X_split)
        guardar_clusters(
            exp_final, ALGORITMO, datos_elegidos[split]["fragment_id"],
            split, etiquetas_split, "predict",
        )
        sil_split = silhouette(X_split, etiquetas_split)
        if np.isfinite(sil_split):
            guardar_metrica(
                exp_final, ALGORITMO, split, "silhouette", sil_split,
                nota=f"predict; {espacio_nota(mejor_receta)}",
            )
            print(f"\nSilhouette en {split}: {sil_split:.4f}")
        interpretacion_externa(etiquetas_split, datos_elegidos[split], split)

    interpretacion_externa(elegido["etiquetas"], datos_elegidos["train"], "train")


if __name__ == "__main__":
    main()
