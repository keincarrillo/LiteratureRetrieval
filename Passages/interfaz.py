"""
Interfaz de consulta de la Etapa 2 (línea de comandos).

Muestra, para cualquier pasaje, su cluster (K-Means y Agglomerative) y sus pesos de temas
(LDA y NMF), y permite revisar las palabras y pasajes representativos y las métricas
de las configuraciones seleccionadas.

Entrada : TSV/documents.tsv y los entregables de TSV/Results/ (generados por results.py)
Uso     :
    python interfaz.py pasaje 432            cluster y temas de un pasaje
    python interfaz.py resumen               clusters y temas de las 4 configuraciones elegidas
    python interfaz.py resumen clustering    solo clustering   (o: temas)
    python interfaz.py metricas              métricas de las 4 configuraciones elegidas
    python interfaz.py                       modo interactivo (menú)

Las configuraciones elegidas se leen de experiments.tsv (columna selected = yes).
"""
import sys
import textwrap
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RES = BASE / "TSV" / "Results"
ANCHO = 90
CLUSTERING = ["K-Means", "Agglomerative"]
TEMAS = ["LDA", "NMF"]


# Carga de datos
def leer(nombre):
    ruta = RES / f"{nombre}.tsv"
    if not ruta.exists():
        sys.exit(f"No se encontró {ruta}. Ejecuta primero los algoritmos y python results.py")
    return pd.read_csv(ruta, sep="\t", dtype=str)


docs = pd.read_csv(BASE / "TSV" / "documents.tsv", sep="\t", dtype={"fragment_id": str}).set_index("fragment_id")
experimentos = leer("experiments")
elegidos = (experimentos[experimentos["selected"] == "yes"]
            .drop_duplicates("method").set_index("method")["experiment_id"].to_dict())
clusters = leer("clustering_results")
resumen_clusters = leer("cluster_summary")
temas = leer("topic_results")
palabras_temas = leer("topic_words")
metricas = leer("metrics")


# Utils
def linea(c="─"):
    print(c * ANCHO)


def titulo(texto):
    print()
    linea("═")
    print(texto)
    linea("═")


def envolver(texto, sangria="    "):
    print(textwrap.fill(texto, ANCHO, initial_indent=sangria, subsequent_indent=sangria))


def obra(fid):
    fid = str(fid).strip()
    return docs.loc[fid, "title"] if fid in docs.index else "?"


def pasajes_con_obra(lista):
    ids = [x.strip() for x in str(lista).split(",") if x.strip()]
    return ", ".join(f"{i} ({obra(i)})" for i in ids)


def palabras_de(tabla, exp, col_id, valor, n=10):
    fila = tabla[(tabla["experiment_id"] == exp) & (tabla[col_id] == str(valor))]
    if fila.empty:
        return ""
    return ", ".join(fila.iloc[0]["top_words"].split(", ")[:n])


# Comando: pasaje <fragment_id>
def cmd_pasaje(fid):
    fid = str(fid).strip()
    if fid not in docs.index:
        ids = docs.index.astype(int)
        print(f"No existe el fragmento {fid}. Los fragment_id van de {ids.min()} a {ids.max()}.")
        return
    d = docs.loc[fid]
    titulo(f"Fragmento {fid} · {d['title']} ({d['author']}) · partición: {d['split']}")
    print(f"  Capítulo: {d['chapter/position']}")
    envolver(" ".join(str(d["text"]).split()[:60]) + " …", "  ")

    print("\n  AGRUPAMIENTO")
    linea()
    for alg in CLUSTERING:
        exp = elegidos.get(alg)
        fila = clusters[(clusters["experiment_id"] == exp) & (clusters["fragment_id"] == fid)]
        if fila.empty:
            print(f"  {alg} ({exp}): sin asignación para este fragmento")
            continue
        c = fila.iloc[0]
        print(f"  {alg} ({exp}): cluster {c['cluster_id']}   [{c['assignment_note']}]")
        envolver("palabras del cluster: " + palabras_de(resumen_clusters, exp, "cluster_id", c["cluster_id"]))

    print("\n  TEMAS (3 de mayor peso)")
    linea()
    for alg in TEMAS:
        exp = elegidos.get(alg)
        filas = temas[(temas["experiment_id"] == exp) & (temas["fragment_id"] == fid)].copy()
        if filas.empty:
            print(f"  {alg} ({exp}): sin pesos para este fragmento")
            continue
        filas["topic_weight"] = filas["topic_weight"].astype(float)
        aviso = "proporciones (suman 1)" if alg == "LDA" else "intensidades (no son probabilidades)"
        print(f"  {alg} ({exp}) — {aviso}")
        for _, t in filas.sort_values("topic_weight", ascending=False).head(3).iterrows():
            pal = palabras_de(palabras_temas, exp, "topic_id", t["topic_id"], n=6)
            print(f"    tema {t['topic_id']:>2}: {t['topic_weight']:.3f}   {pal}")
    print()


# Comando: resumen [clustering | temas]
def cmd_resumen(tarea=None):
    if tarea in (None, "clustering"):
        for alg in CLUSTERING:
            exp = elegidos.get(alg)
            titulo(f"{alg} ({exp}) — clusters, palabras y pasajes representativos")
            tabla = resumen_clusters[resumen_clusters["experiment_id"] == exp]
            for _, c in tabla.sort_values("cluster_id", key=lambda s: s.astype(int)).iterrows():
                print(f"  Cluster {c['cluster_id']} · {c['size']} fragmentos")
                envolver("palabras: " + c["top_words"])
                envolver("pasajes: " + pasajes_con_obra(c["representative_passages"]))
    if tarea in (None, "temas"):
        for alg in TEMAS:
            exp = elegidos.get(alg)
            titulo(f"{alg} ({exp}) — temas, palabras y pasajes representativos")
            tabla = palabras_temas[palabras_temas["experiment_id"] == exp]
            for _, t in tabla.sort_values("topic_id", key=lambda s: s.astype(int)).iterrows():
                print(f"  Tema {t['topic_id']}")
                envolver("palabras: " + t["top_words"])
                envolver("pasajes: " + pasajes_con_obra(t["representative_passages"]))
    print()


# Comando: metricas
def cmd_metricas():
    titulo("Configuraciones elegidas y sus métricas")
    for alg in CLUSTERING + TEMAS:
        exp = elegidos.get(alg)
        e = experimentos[experimentos["experiment_id"] == exp].iloc[0]
        print(f"  {alg} ({exp}) · receta {e['recipe']} · {e['normalization']} · {e['representation']} · "
              f"{e['ngram_range']} · {e['vocabulary']} · {e['dim_reduction']}")
        envolver("hiperparámetros: " + str(e["hyperparameters"]))
        m = metricas[metricas["experiment_id"] == exp]
        for _, r in m.iterrows():
            nota = f" — {r['note']}" if isinstance(r["note"], str) and r["note"] else ""
            interp = f"  ({r['interpretation']})" if isinstance(r["interpretation"], str) else ""
            print(f"    {r['split']:>5} · {r['metric']}: {float(r['value']):.4f}{interp}{nota}")
        linea()
    print("  Nota: silhouette, perplejidad y error de reconstrucción no se comparan entre sí.")
    print(f"  Experimentos registrados en experiments.tsv: {len(experimentos)}\n")


# Menú interactivo
def menu():
    print("\nInterfaz Etapa 2 — Pasajes")
    print("  Configuraciones elegidas: " + ", ".join(f"{a} {e}" for a, e in elegidos.items()))
    while True:
        print("\n  1) Ver un pasaje   2) Resumen de clusters   3) Resumen de temas   4) Métricas   0) Salir")
        op = input("  Opción: ").strip()
        if op == "1":
            cmd_pasaje(input("  fragment_id: "))
        elif op == "2":
            cmd_resumen("clustering")
        elif op == "3":
            cmd_resumen("temas")
        elif op == "4":
            cmd_metricas()
        elif op == "0":
            break


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        menu()
    elif args[0] == "pasaje" and len(args) == 2:
        cmd_pasaje(args[1])
    elif args[0] == "resumen":
        cmd_resumen(args[1] if len(args) > 1 else None)
    elif args[0] == "metricas":
        cmd_metricas()
    else:
        print(__doc__)