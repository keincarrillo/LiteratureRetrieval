from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from representation import armar_vectorizador, cargar_textos, recetas_para
from results import COLUMNAS

SEED, K, ELEGIDO = 3, 8, "N19"
SEMILLAS = (3, 11, 29)
SALIDA = Path(__file__).resolve().parent / "TSV" / "Results" / "nmf"
RECETAS = recetas_para("nmf")
BASE = dict(n_components=K, init="nndsvda", max_iter=1500, tol=1e-4,
            random_state=SEED, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)

# ID, receta, parametros diferentes respecto a BASE.
EXPERIMENTOS = [
    ("N01", "norm_N4", {"n_components": 3}),
    ("N02", "norm_N4", {"n_components": 5}),
    ("N03", "norm_N4", {}),
    ("N04", "norm_N4", {"n_components": 10}),
    *[(f"N{i:02d}", receta, {}) for i, receta in enumerate(RECETAS, 5)],
    ("N14", "norm_N4", {}),
    ("N15", "norm_N4", {"alpha_W": 0.0001}),
    ("N16", "norm_N4", {"alpha_W": 0.001}),
    ("N17", "norm_N4", {"alpha_H": 0.0001}),
    ("N18", "norm_N4", {"alpha_H": 0.001}),
    ("N19", "norm_N4", {"alpha_W": 0.0001, "alpha_H": 0.0001}),
    ("N20", "norm_N4", {"alpha_W": 0.0001, "alpha_H": 0.0001, "l1_ratio": 0.5}),
    ("N21", "norm_N4", {"alpha_W": 0.0001, "alpha_H": 0.0001, "l1_ratio": 1.0}),
    ("N22", "norm_N4", {"init": "random"}),
    ("N23", "norm_N4", {"init": "nndsvdar"}),
    ("N24", "norm_N4", {"max_iter": 50}),
    ("N25", "norm_N4", {"max_iter": 300}),
]

# Estos nombres proceden de versiones anteriores; no son entregables oficiales.
COMPLEMENTARIOS = (
    "resumen_nmf_k.tsv", "resumen_nmf_recetas.tsv",
    "resumen_nmf_hiperparametros.tsv", "diagnostico_nmf.tsv",
    "diagnostico_temas_nmf.tsv", "diagnostico_estabilidad_nmf.tsv",
    "resumen_nmf_final.tsv", "temas_nmf_final.tsv",
    "distribucion_nmf_test.tsv",
)


def preparar(nombre, cache):
    """Ajusta cada vectorizador una sola vez, exclusivamente con TRAIN."""
    if nombre not in cache:
        receta = RECETAS[nombre]
        datos = cargar_textos(receta["norm"])
        vec = armar_vectorizador(receta)
        x_train = vec.fit_transform(datos["train"]["texto"])
        x_dev = vec.transform(datos["dev"]["texto"])
        cache[nombre] = (datos, vec, x_train, x_dev)
    return cache[nombre]


def entrenar(x, hp):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo = NMF(**hp)
        w_train = modelo.fit_transform(x)
    convergio = modelo.n_iter_ < hp["max_iter"] and not any(
        issubclass(a.category, ConvergenceWarning) for a in avisos
    )
    return modelo, w_train, convergio


def error(x, w, h):
    """Norma ||X-WH|| de Frobenius, conservando X dispersa."""
    norma_x = float(x.multiply(x).sum())
    norma_wh = float(np.sum((w @ (h @ h.T)) * w))
    cruzado = float(np.sum(w * (x @ h.T)))
    return float(np.sqrt(max(0.0, norma_x + norma_wh - 2 * cruzado)))


def guardar_pesos(bloques, exp, split, ids, w):
    n, k = w.shape
    bloques.append(pd.DataFrame({
        "experiment_id": exp,
        "algorithm": "NMF",
        "fragment_id": np.repeat(ids.to_numpy(), k),
        "split": split,
        "topic_id": np.tile(np.arange(k), n),
        "topic_weight": np.maximum(w.ravel(), 0).round(6),
    }))


def temas_libros(df, w, k):
    validos = w.max(axis=1) > 1e-10
    tabla = pd.crosstab(df.loc[validos, "book_id"], w[validos].argmax(axis=1))
    tabla = tabla.reindex(columns=range(k), fill_value=0)
    return tabla, int(((tabla > 0).sum(axis=0) > 1).sum())


class Coherencia:
    """NPMI con coocurrencia por fragmento en TRAIN-N3, referencia comun."""

    def __init__(self):
        textos = cargar_textos("N3")["train"]["texto"]
        vec = CountVectorizer(binary=True, tokenizer=str.split,
                              token_pattern=None, lowercase=False)
        self.x = vec.fit_transform(textos)
        self.vocab = vec.vocabulary_
        self.n = self.x.shape[0]
        self.presencias = {}

    def npmi(self, palabras):
        for palabra in palabras:
            if palabra not in self.presencias:
                indice = self.vocab.get(palabra)
                self.presencias[palabra] = (
                    np.zeros(self.n, dtype=np.int32) if indice is None
                    else self.x[:, indice].toarray().ravel().astype(np.int32)
                )
        p = np.column_stack([self.presencias[t] for t in palabras])
        f = p.sum(axis=0)
        juntos = p.T @ p
        valores = []
        for i in range(len(palabras)):
            for j in range(i + 1, len(palabras)):
                n_ij = juntos[i, j]
                if not n_ij or not f[i] or not f[j]:
                    valores.append(-1.0)
                    continue
                pi, pj, pij = f[i] / self.n, f[j] / self.n, n_ij / self.n
                valores.append(1.0 if pij == 1 else
                                np.log(pij / (pi * pj)) / -np.log(pij))
        return float(np.mean(valores))


def diagnosticar(candidatos, filas_experimentos):
    print("\n=== COHERENCIA Y ESTABILIDAD (TRAIN/DEV) ===")
    evaluador = Coherencia()
    for exp in ("N05", "N14", "N19"):
        datos, vec, x_train, modelo, hp, conv, w_dev = candidatos[exp]
        palabras = vec.get_feature_names_out()
        h = modelo.components_
        npmis = []
        for tema in range(K):
            top = palabras[np.argsort(h[tema])[-10:][::-1]].tolist()
            npmis.append(evaluador.npmi(top))
        estabilidad = []
        for semilla in SEMILLAS:
            alternativo, _, converge = entrenar(
                x_train, {**hp, "init": "random", "random_state": semilla}
            )
            if converge:
                similitud = cosine_similarity(h, alternativo.components_)
                filas, columnas = linear_sum_assignment(-similitud)
                estabilidad.append(float(similitud[filas, columnas].mean()))
        _, compartidos = temas_libros(datos["dev"], w_dev, K)
        media_npmi = float(np.mean(npmis))
        media_est = float(np.mean(estabilidad)) if estabilidad else float("nan")
        print(f"{exp}: NPMI={media_npmi:.4f}, estabilidad={media_est:.4f} "
              f"({len(estabilidad)}/3), compartidos={compartidos}, "
              f"convergio={conv}")
        for fila in filas_experimentos:
            if fila["experiment_id"] == exp:
                fila["notes"] += (f"; npmi_train={media_npmi:.5f}; "
                                  f"estabilidad={media_est:.5f}")


def main():
    if len(EXPERIMENTOS) != 25 or [e[0] for e in EXPERIMENTOS] != [
        f"N{i:02d}" for i in range(1, 26)
    ]:
        raise ValueError("Las recetas cambiaron: revisar IDs N01-N25")

    cache, ajustes, candidatos = {}, {}, {}
    filas = {nombre: [] for nombre in ("experiments", "metrics", "topic_words")}
    pesos = []

    print("=== 25 EXPERIMENTOS NMF (solo TRAIN/DEV) ===")
    for exp, nombre, cambios in EXPERIMENTOS:
        datos, vec, x_train, x_dev = preparar(nombre, cache)
        hp = {**BASE, **cambios}
        llave = (nombre, tuple(sorted(hp.items())))
        if llave not in ajustes:
            modelo, w_train, conv = entrenar(x_train, hp)
            ajustes[llave] = (modelo, w_train, modelo.transform(x_dev), conv)
        modelo, w_train, w_dev, conv = ajustes[llave]
        h, k = modelo.components_, hp["n_components"]
        e_train = float(modelo.reconstruction_err_)
        e_dev = error(x_dev, w_dev, h)
        _, compartidos = temas_libros(datos["dev"], w_dev, k)
        receta, ng = RECETAS[nombre], RECETAS[nombre]["ngram"]

        filas["experiments"].append(dict(
            experiment_id=exp, task="topics", method="NMF", recipe=nombre,
            normalization=receta["norm"], representation="TF-IDF",
            ngram_range=f"({ng[0]},{ng[1]})", vocabulary=receta["vocab"],
            dim_reduction="none",
            hyperparameters="; ".join(f"{p}={v}" for p, v in hp.items()),
            dev_metric="reconstruction_error", dev_value=round(e_dev, 6),
            selected="yes" if exp == ELEGIDO else "no",
            notes=(f"iteraciones={modelo.n_iter_}; convergio={conv}; "
                   f"temas_compartidos={compartidos}"),
        ))
        for split, valor in (("train", e_train), ("dev", e_dev)):
            filas["metrics"].append(dict(
                experiment_id=exp, algorithm="NMF", split=split,
                metric="reconstruction_error", value=round(valor, 6),
                interpretation="lower is better", note="TF-IDF fit solo TRAIN",
            ))
        guardar_pesos(pesos, exp, "train", datos["train"]["fragment_id"], w_train)
        guardar_pesos(pesos, exp, "dev", datos["dev"]["fragment_id"], w_dev)

        vocab = vec.get_feature_names_out()
        for tema in range(k):
            top = vocab[np.argsort(h[tema])[-10:][::-1]].tolist()
            mejores = [i for i in np.argsort(w_dev[:, tema])[::-1]
                       if w_dev[i, tema] > 1e-10][:2]
            pasajes = [int(datos["dev"].iloc[i]["fragment_id"]) for i in mejores]
            filas["topic_words"].append(dict(
                experiment_id=exp, algorithm="NMF", topic_id=tema,
                top_words=", ".join(top),
                representative_passages=", ".join(map(str, pasajes)),
            ))

        if exp in ("N05", "N14", "N19"):
            candidatos[exp] = (datos, vec, x_train, modelo, hp, conv, w_dev)
        print(f"{exp} {nombre:10} k={k:2} iter={modelo.n_iter_:4} "
              f"convergio={conv!s:5} dev={e_dev:.4f}", flush=True)

    diagnosticar(candidatos, filas["experiments"])

    # N19 ya fue seleccionado mediante DEV: TEST solo se transforma ahora.
    datos, vec, _, modelo, _, convergio, _ = candidatos[ELEGIDO]
    if not convergio:
        raise RuntimeError("N19 no convergio; no se evaluara TEST")
    x_test = vec.transform(datos["test"]["texto"])
    w_test = modelo.transform(x_test)
    e_test = error(x_test, w_test, modelo.components_)
    guardar_pesos(pesos, ELEGIDO, "test", datos["test"]["fragment_id"], w_test)
    filas["metrics"].append(dict(
        experiment_id=ELEGIDO, algorithm="NMF", split="test",
        metric="reconstruction_error", value=round(e_test, 6),
        interpretation="lower is better", note="evaluacion final: TEST",
    ))
    tabla, _ = temas_libros(datos["test"], w_test, K)
    print(f"\nN19 TEST: error={e_test:.4f}, "
          f"error/raiz(n)={e_test / np.sqrt(len(datos['test'])):.4f}")
    print(tabla)

    # Las columnas siguen exactamente el contrato de results.py.
    salida = {n: pd.DataFrame(filas[n], columns=COLUMNAS[n])
              for n in filas}
    salida["topic_results"] = pd.concat(pesos, ignore_index=True)[
        COLUMNAS["topic_results"]
    ]
    esperados = {
        "experiments": len(EXPERIMENTOS),
        "metrics": len(EXPERIMENTOS) * 2 + 1,
        "topic_words": sum(({**BASE, **c}["n_components"])
                           for _, _, c in EXPERIMENTOS),
    }
    esperados["topic_results"] = (esperados["topic_words"] *
                                  (len(datos["train"]) + len(datos["dev"]))
                                  + K * len(datos["test"]))
    claves = {
        "experiments": ["experiment_id"],
        "metrics": ["experiment_id", "split", "metric"],
        "topic_words": ["experiment_id", "topic_id"],
        "topic_results": ["experiment_id", "split", "fragment_id", "topic_id"],
    }
    for nombre, df in salida.items():
        if len(df) != esperados[nombre] or df.duplicated(claves[nombre]).any():
            raise RuntimeError(f"Registros incorrectos o duplicados: {nombre}")
    if salida["experiments"].loc[
        salida["experiments"]["selected"] == "yes", "experiment_id"
    ].tolist() != [ELEGIDO]:
        raise RuntimeError("Debe seleccionarse solamente N19")

    SALIDA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SALIDA.parent, prefix="nmf_tmp_") as tmp:
        for nombre, df in salida.items():
            temporal = Path(tmp) / f"{nombre}.tsv"
            df.to_csv(temporal, sep="\t", index=False)
        for nombre in salida:
            (Path(tmp) / f"{nombre}.tsv").replace(SALIDA / f"{nombre}.tsv")
    # Solo borra los 9 complementarios conocidos de versiones anteriores.
    for nombre in COMPLEMENTARIOS:
        (SALIDA / nombre).unlink(missing_ok=True)

    print("\n=== ARCHIVOS LISTOS PARA INTEGRACION ===")
    for nombre in ("experiments", "metrics", "topic_results", "topic_words"):
        print(f"{nombre}.tsv: {len(salida[nombre])} filas OK")
    print("N19 seleccionado; TEST solamente en N19.")


if __name__ == "__main__":
    main()
