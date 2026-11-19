"""
Integrante 4 - LDA (Latent Dirichlet Allocation) sobre fragmentos literarios.

Flujo (README, sección 3):
    1. Probar las 9 recetas de recetas_para("lda") con hiperparámetros por defecto.
       Como la perplejidad NO es comparable entre recetas (cambia el vocabulario),
       la mejor receta se elige con coherencia NPMI + diversidad de temas (ver abajo).
    2. Con la receta fija, búsqueda pequeña de hiperparámetros; se elige con la
       perplejidad en dev (el vocabulario ya es el mismo en todas las corridas).
    3. Interpretar: top 10 palabras, pasajes representativos, tema dominante x book_id.
    4. Test: la configuración elegida se evalúa UNA sola vez.
    5. Guardar todo con results.py.

Salida: TSV/Results/lda/   (experiments, metrics, topic_results, topic_words)
Ejecutar: python lda.py
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline

from representation import BASE, SEED, armar_vectorizador, cargar_textos, recetas_para
from results import (
    guardar_experimento,
    guardar_metrica,
    guardar_palabras_tema,
    guardar_temas,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ALGORITMO = "LDA"
TOP_N = 10          # palabras por tema que se reportan
N_PASAJES = 3       # pasajes representativos por tema (el enunciado pide >= 2)
MIN_DOMINANTE = 5   # un tema debe ser el dominante de >= 5 fragmentos de train (si no, está "vacío")
MAX_FRAC = 0.5      # ningún tema puede ser el dominante de más de la mitad de train (tema "gigante")

# Valores por defecto de scikit-learn, usados en el paso 1 (comparar recetas)
HP_DEFECTO = dict(n_components=10, max_iter=10, learning_method="batch",
                  doc_topic_prior=None, topic_word_prior=None)

# Búsqueda pequeña del paso 2 (README, sección 6)
K_OPCIONES = [5, 8, 10, 15]
LEARNING_OPCIONES = ["batch", "online"]
DOC_PRIOR_OPCIONES = [None, 0.1]
TOPIC_PRIOR_OPCIONES = [None, 0.01]
MAX_ITER_BUSQUEDA = 20           # etapa A: se fija en 20 y se explora la rejilla
MAX_ITER_OPCIONES = [10, 50]     # etapa B: con la mejor de A se prueban 10 y 50 (20 ya está)


# Utilidades
def hp_texto(hp):
    """Hiperparámetros listos para experiments.tsv (None se escribe como '1/k')."""
    d = dict(hp)
    d["k"] = d.pop("n_components")
    if d["doc_topic_prior"] is None:
        d["doc_topic_prior"] = "1/k"
    if d["topic_word_prior"] is None:
        d["topic_word_prior"] = "1/k"
    d["seed"] = SEED
    return d


def vectores_vacios(X):
    """Índices de fragmentos sin ninguna palabra tras el vectorizador (X.sum(axis=1) == 0)."""
    return np.where(np.asarray(X.sum(axis=1)).ravel() == 0)[0]


def perplejidad(lda, X):
    """Perplejidad de LDA sobre X (matriz de conteos). Excluye documentos vacíos."""
    vacios = vectores_vacios(X)
    if len(vacios):
        X = X[np.setdiff1d(np.arange(X.shape[0]), vacios)]
    return float(lda.perplexity(X))


def top_palabras_por_tema(lda, vocab, n=TOP_N):
    """Top n palabras (mayor peso) de cada tema, en el espacio original de palabras."""
    return [vocab[fila.argsort()[::-1][:n]].tolist() for fila in lda.components_]


def coherencia_npmi(tops, X_ref):
    """
    Coherencia NPMI promedio: para cada tema, promedio del NPMI de todos los pares de
    sus top 10 palabras, con coocurrencia a nivel documento en X_ref (matriz de
    conteos de TRAIN, con el vocabulario de la receta). Rango [-1, 1], mayor = mejor.
    Un par que nunca coocurre vale -1. Es una métrica auxiliar: no demuestra por sí
    sola la calidad semántica.
    """
    B = (X_ref > 0).astype(np.float64).tocsc()
    n_docs = B.shape[0]
    df = np.asarray(B.sum(axis=0)).ravel()
    eps = 1e-12
    valores = []
    for palabras, idx in tops:
        npmis = []
        for i, j in itertools.combinations(idx, 2):
            p_i, p_j = df[i] / n_docs, df[j] / n_docs
            p_ij = B[:, i].multiply(B[:, j]).sum() / n_docs
            if p_ij == 0:
                npmis.append(-1.0)
            else:
                npmis.append(np.log((p_ij + eps) / (p_i * p_j)) / -np.log(p_ij + eps))
        valores.append(np.mean(npmis))
    return float(np.mean(valores))


def diversidad(tops_palabras):
    """Proporción de palabras distintas entre los top 10 de todos los temas (1 = sin redundancia)."""
    todas = [w for t in tops_palabras for w in t]
    return len(set(todas)) / len(todas)


def calidad_temas(lda, vec, X_train):
    """(coherencia NPMI, diversidad) de un LDA ya ajustado."""
    vocab = vec.get_feature_names_out()
    idx = [fila.argsort()[::-1][:TOP_N] for fila in lda.components_]
    tops = [(vocab[i].tolist(), i) for i in idx]
    return coherencia_npmi(tops, X_train), diversidad([t[0] for t in tops])


def validos(resultados):
    """Configuraciones sin temas colapsados; si ninguna lo logra, se avisa y se usan todas."""
    ok = [r for r in resultados if r["valido"]]
    if not ok:
        print("  AVISO: ninguna configuración evita el colapso de temas; se usan todas.")
        return resultados
    return ok


def ajustar(vec, X_train, hp):
    """Ajusta LDA (solo con train) sobre la matriz de conteos ya vectorizada."""
    lda = LatentDirichletAllocation(random_state=SEED, **hp)
    lda.fit(X_train)
    return lda


def correr_experimento(exp_id, nombre_receta, receta, vec, mats, hp, seleccionado=False, extra=""):
    """Ajusta con train, mide perplejidad train/dev y calidad de temas, y guarda el experimento."""
    lda = ajustar(vec, mats["train"], hp)
    p_train = perplejidad(lda, mats["train"])
    p_dev = perplejidad(lda, mats["dev"])
    npmi, div = calidad_temas(lda, vec, mats["train"])

    # Chequeo de colapso: cuántos fragmentos de train tienen a cada tema como dominante
    dom = np.bincount(lda.transform(mats["train"]).argmax(axis=1), minlength=hp["n_components"])
    min_dom, max_frac = int(dom.min()), float(dom.max() / dom.sum())
    valido = min_dom >= MIN_DOMINANTE and max_frac <= MAX_FRAC

    nota = (f"perp_train={p_train:.1f} | npmi={npmi:.3f} | diversidad={div:.2f} | "
            f"min_dominante={min_dom} | max_frac={max_frac:.2f}")
    if not valido:
        nota += " | DESCARTADO: temas colapsados (vacios o gigante)"
    if extra:
        nota += f" | {extra}"
    guardar_experimento(exp_id, ALGORITMO, nombre_receta, receta, hp_texto(hp),
                        "perplexity", round(p_dev, 6), seleccionado=seleccionado, notas=nota)
    guardar_metrica(exp_id, ALGORITMO, "train", "perplexity", p_train,
                    nota=f"fixed vocabulary ({nombre_receta})")
    guardar_metrica(exp_id, ALGORITMO, "dev", "perplexity", p_dev,
                    nota=f"fixed vocabulary ({nombre_receta})")
    guardar_metrica(exp_id, ALGORITMO, "train", "coherence_npmi", npmi,
                    nota="top 10 words, document co-occurrence on train; higher is better")
    return dict(exp_id=exp_id, nombre_receta=nombre_receta, receta=receta, hp=hp, lda=lda,
                p_train=p_train, p_dev=p_dev, npmi=npmi, div=div,
                min_dom=min_dom, max_frac=max_frac, valido=valido)


# Reporte e interpretación
def reportar_temas(res, vec, datos, mats, textos_originales):
    """Top 10 palabras y pasajes representativos de cada tema; los guarda con results.py."""
    exp_id, lda = res["exp_id"], res["lda"]
    vocab = vec.get_feature_names_out()
    tops = top_palabras_por_tema(lda, vocab)
    pesos_train = lda.transform(mats["train"])          # proporciones: cada fila suma 1
    ids_train = datos["train"]["fragment_id"].to_numpy()
    libros = datos["train"]["book_id"].to_numpy()
    dominante = pesos_train.argmax(axis=1)

    print(f"\nTemas de {exp_id} (receta {res['nombre_receta']}):")
    for t, palabras in enumerate(tops):
        orden = pesos_train[:, t].argsort()[::-1][:N_PASAJES]     # mayor peso del tema
        pasajes = ids_train[orden].tolist()
        guardar_palabras_tema(exp_id, ALGORITMO, t, palabras, pasajes)

        n_dom = int((dominante == t).sum())
        libros_t = pd.Series(libros[dominante == t]).value_counts().to_dict()
        print(f"  Tema {t:2d} (dominante en {n_dom:3d} train) {', '.join(palabras)}")
        print(f"           pasajes={pasajes} | libros={libros_t}")
        for fid, w in zip(pasajes[:2], pesos_train[orden[:2], t]):
            fragmento = textos_originales.get(fid, "")[:140].replace("\n", " ")
            print(f"             [{fid}] peso={w:.2f}: {fragmento}...")


def interpretacion_externa(lda, mats, datos):
    """Tema dominante x book_id (solo interpretación; book_id nunca entra al modelo)."""
    for split in ("train", "dev", "test"):
        if split not in mats:
            continue
        dom = lda.transform(mats[split]).argmax(axis=1)
        tabla = pd.crosstab(pd.Series(dom, name="tema"), datos[split]["book_id"])
        print(f"\nTema dominante x book_id ({split}):")
        print(tabla.to_string())


def main():
    recetas = recetas_para("lda")
    cache_datos, cache_mats = {}, {}

    def preparar(nombre, receta):
        """Textos y matrices de conteos de una receta; el vectorizador se ajusta solo con train."""
        if nombre in cache_mats:
            return cache_datos[nombre], cache_mats[nombre]
        datos = cargar_textos(receta["norm"])
        vec = armar_vectorizador(receta)
        mats = {"train": vec.fit_transform(datos["train"]["texto"])}
        mats["dev"] = vec.transform(datos["dev"]["texto"])
        mats["test"] = vec.transform(datos["test"]["texto"])
        cache_datos[nombre], cache_mats[nombre] = datos, (vec, mats)
        return datos, (vec, mats)

    # Documentos vacíos (X.sum(axis=1) == 0) en cada receta
    print("=== Documentos vacíos tras el vectorizador ===")
    for nombre, receta in recetas.items():
        _, (_, mats) = preparar(nombre, receta)
        vacios = {s: len(vectores_vacios(m)) for s, m in mats.items()}
        print(f"  {nombre:10s} {vacios}")
    print("  Manejo: se excluyen de la perplejidad; en transform() recibirían la distribución "
          "a priori (casi uniforme), por lo que se avisaría en el reporte.")

    # 1. Comparar las 9 recetas con hiperparámetros por defecto
    print("\n=== Paso 1: 9 recetas con hiperparámetros por defecto ===")
    print("(perplejidad NO comparable entre recetas; se elige por coherencia NPMI + diversidad)")
    num_exp = 1
    resultados_recetas = []
    for nombre, receta in recetas.items():
        _, (vec, mats) = preparar(nombre, receta)
        exp_id = f"L{num_exp:02d}"
        num_exp += 1
        res = correr_experimento(exp_id, nombre, receta, vec, mats, HP_DEFECTO,
                                 extra="receta con hiperparametros por defecto")
        resultados_recetas.append(res)
        print(f"  {exp_id} {nombre:10s} vocab={mats['train'].shape[1]:6d} "
              f"perp_dev={res['p_dev']:9.1f} npmi={res['npmi']:.3f} div={res['div']:.2f}")

    mejor_receta_res = max(resultados_recetas, key=lambda r: (r["npmi"] * r["div"]))
    mejor_nombre = mejor_receta_res["nombre_receta"]
    mejor_receta = mejor_receta_res["receta"]
    datos, (vec, mats) = preparar(mejor_nombre, mejor_receta)
    print(f"\nMejor receta: {mejor_nombre} ({mejor_receta_res['exp_id']}, "
          f"npmi={mejor_receta_res['npmi']:.3f}, div={mejor_receta_res['div']:.2f})")

    # 2. Búsqueda pequeña de hiperparámetros con la receta fija
    print(f"\n=== Paso 2: hiperparámetros sobre {mejor_nombre} (perplejidad en dev) ===")
    resultados_hp = []

    def probar(hp):
        nonlocal num_exp
        exp_id = f"L{num_exp:02d}"
        num_exp += 1
        res = correr_experimento(exp_id, mejor_nombre, mejor_receta, vec, mats, hp)
        resultados_hp.append(res)
        print(f"  {exp_id} k={hp['n_components']:2d} it={hp['max_iter']:2d} {hp['learning_method']:6s} "
              f"doc={str(hp['doc_topic_prior']):4s} word={str(hp['topic_word_prior']):4s} "
              f"perp_train={res['p_train']:8.1f} perp_dev={res['p_dev']:8.1f} "
              f"npmi={res['npmi']:.3f} div={res['div']:.2f} "
              f"mín_dom={res['min_dom']:3d} máx={res['max_frac']:.2f}"
              f"{'' if res['valido'] else ' [descartado]'}")
        return res

    # Etapa A: n_components x learning_method x doc_topic_prior x topic_word_prior
    for k, lm, dp, tp in itertools.product(K_OPCIONES, LEARNING_OPCIONES,
                                           DOC_PRIOR_OPCIONES, TOPIC_PRIOR_OPCIONES):
        probar(dict(n_components=k, max_iter=MAX_ITER_BUSQUEDA, learning_method=lm,
                    doc_topic_prior=dp, topic_word_prior=tp))

    # Etapa B: con la mejor de A, variar max_iter
    mejor_A = min(validos(resultados_hp), key=lambda r: r["p_dev"])
    print(f"\nMejor de la etapa A: {mejor_A['exp_id']} -> variando max_iter")
    for it in MAX_ITER_OPCIONES:
        probar({**mejor_A["hp"], "max_iter": it})

    # Elegir con dev (menor perplejidad)
    elegido = min(validos(resultados_hp), key=lambda r: r["p_dev"])
    exp_final, lda = elegido["exp_id"], elegido["lda"]
    print(f"\nConfiguración elegida: {exp_final} ({mejor_nombre}, {hp_texto(elegido['hp'])}, "
          f"perp_dev={elegido['p_dev']:.1f})")

    # Marcar el experimento elegido como seleccionado=True
    guardar_experimento(
        exp_final, ALGORITMO, mejor_nombre, mejor_receta, hp_texto(elegido["hp"]),
        "perplexity", round(elegido["p_dev"], 6), seleccionado=True,
        notas=(f"perp_train={elegido['p_train']:.1f} | npmi={elegido['npmi']:.3f} | "
               f"diversidad={elegido['div']:.2f} | configuracion elegida (menor perplejidad en dev entre configuraciones sin temas colapsados)"))

    # 3. Interpretar: temas, pasajes representativos, tema x book_id
    docs = pd.read_csv(BASE / "TSV" / "documents.tsv", sep="\t", usecols=["fragment_id", "text"])
    textos_originales = dict(zip(docs["fragment_id"], docs["text"]))
    reportar_temas(elegido, vec, datos, mats, textos_originales)

    # Pesos de tema (proporciones, cada fila suma 1) de train y dev
    for split in ("train", "dev"):
        guardar_temas(exp_final, ALGORITMO, datos[split]["fragment_id"], split,
                      lda.transform(mats[split]))

    # 4. Test: UNA sola vez, con la configuración ya elegida
    p_test = perplejidad(lda, mats["test"])
    guardar_metrica(exp_final, ALGORITMO, "test", "perplexity", p_test,
                    nota=f"fixed vocabulary ({mejor_nombre}); single final evaluation")
    guardar_temas(exp_final, ALGORITMO, datos["test"]["fragment_id"], "test",
                  lda.transform(mats["test"]))
    print(f"\nPerplejidad final: train={elegido['p_train']:.1f}  dev={elegido['p_dev']:.1f}  "
          f"test={p_test:.1f}")
    print("(dev/test >> train es esperado: corpus pequeño y vocabulario ajustado con train)")

    interpretacion_externa(lda, mats, datos)


if __name__ == "__main__":
    main()
