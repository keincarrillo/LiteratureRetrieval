"""Evaluación del Benchmark y Generación de Métricas - Calcula P@10, R@10, F1@10, MRR y nDCG@10 sobre las 6 configuraciones VSM
(Frecuencia, Binaria y TF-IDF x Unigramas y Bigramas) sobre todo el corpus (500 documentos).

Salidas:
  rankings.tsv                          Ranking oficial (mejor configuración).
  combined_view.tsv                     Ranking oficial con la relevancia de cada documento.
  results/benchmark/combined_view_<cfg>.tsv  Vista combinada de cada configuración.
  results/benchmark/rankings_<cfg>.tsv  Ranking Top-10 de cada una de las 6 configuraciones.
  results/benchmark/metrics_by_query.tsv   Métricas por consulta y configuración.
  results/benchmark/metrics_by_config.tsv  Promedios por configuración.
  results/benchmark/metrics_by_level.tsv   Promedios por nivel y configuración.

python main.py --benchmark
python -m src.benchmark --queries queries.tsv --qrels qrels.tsv --output rankings.tsv
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .nlp import normalize_query
from .vsm import load_model

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "benchmark"
TOP_K = 10

# Configuraciones oficiales (Tabla 9): (representación, n-grama, etiqueta repr., etiqueta n-grama)
CONFIGS = [
    ("frequency", "unigram", "Frecuencia", "Uni"),
    ("frequency", "bigram", "Frecuencia", "Bi"),
    ("binary", "unigram", "Binaria", "Uni"),
    ("binary", "bigram", "Binaria", "Bi"),
    ("tfidf", "unigram", "TF-IDF", "Uni"),
    ("tfidf", "bigram", "TF-IDF", "Bi"),
]
METRICS = ["p10", "r10", "f1", "rr", "ndcg10"]
LEVEL_NAMES = {1: "Fácil", 2: "Media", 3: "Difícil"}


def load_queries(path: Path) -> Dict[str, Dict]:
    """Carga queries.tsv (query_id, difficulty_level, query)."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de consultas: {path}")

    queries = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query_id"].strip()
            level = int(row["difficulty_level"].strip())
            qtext = row["query"].strip()
            queries[qid] = {"level": level, "query": qtext}
    return queries


def load_qrels(path: Path) -> Dict[str, Dict[int, int]]:
    """Carga qrels.tsv (query_id, document_id, relevance)."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de juicios qrels: {path}")

    qrels = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query_id"].strip()
            doc_id = int(row["document_id"].strip())
            rel = int(row["relevance"].strip())
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def evaluate_ranking(top_docs: List[Tuple[int, float]], qrel: Dict[int, int]) -> Dict[str, float]:
    """Calcula P@10, R@10, F1@10, RR y nDCG@10 para un Top-10 de documentos.

    Si top_docs tiene menos de 10 documentos (o ninguno), P@10 se sigue dividiendo entre 10:
    no recuperar un documento cuenta como fallo.
    """
    total_rel_in_benchmark = len(qrel)
    if total_rel_in_benchmark == 0:
        return {m: 0.0 for m in METRICS}

    # P@10 y R@10 (cualquier relevancia > 0 cuenta como relevante)
    rel_retrieved = sum(1 for doc_id, _ in top_docs if doc_id in qrel)
    p10 = rel_retrieved / float(TOP_K)
    r10 = rel_retrieved / float(total_rel_in_benchmark)
    f1 = (2.0 * p10 * r10 / (p10 + r10)) if (p10 + r10) > 0 else 0.0

    # Reciprocal Rank (RR): 1 / rank del primer relevante
    rr = 0.0
    for rank, (doc_id, _) in enumerate(top_docs, start=1):
        if doc_id in qrel:
            rr = 1.0 / rank
            break

    # nDCG@10: ganancia graduada (2^rel - 1) con descuento log2(rank + 1)
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(top_docs, start=1):
        rel = qrel.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(qrel.values(), reverse=True)[:TOP_K]
    idcg = sum((2**r - 1) / math.log2(rank + 1) for rank, r in enumerate(ideal_rels, start=1))
    ndcg10 = (dcg / idcg) if idcg > 0 else 0.0

    return {"p10": p10, "r10": r10, "f1": f1, "rr": rr, "ndcg10": ndcg10}


def rank_documents(q_vec, matrix, doc_ids) -> List[Tuple[int, float]]:
    """Top-10 por similitud coseno con orden estable.

    Siempre devuelve 10 documentos (el evaluador oficial exige ranks 1..10 en cada consulta).
    Si hay menos de 10 documentos con score > 0, el resto son empates en 0 ordenados por
    posición en el corpus: son relleno, no recuperaciones reales.
    """
    sims = cosine_similarity(q_vec, matrix)[0]
    top_indices = np.argsort(-sims, kind="stable")[:TOP_K]
    return [(int(doc_ids[i]), float(sims[i])) for i in top_indices]


def average(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Promedio de cada métrica sobre una lista de resultados por consulta."""
    n = len(metrics_list)
    return {m: (sum(x[m] for x in metrics_list) / n if n else 0.0) for m in METRICS}


def write_rankings(path: Path, eval_qids: List[str], rankings: Dict[str, List[Tuple[int, float]]]):
    """Escribe un ranking en el formato oficial (query_id, document_id, rank, score)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "document_id", "rank", "score"])
        for qid in eval_qids:
            for rank, (doc_id, score) in enumerate(rankings[qid], start=1):
                writer.writerow([qid, doc_id, rank, f"{score:.4f}"])


def write_combined_view(path: Path, eval_qids: List[str], rankings: Dict[str, List[Tuple[int, float]]],
                        qrels: Dict[str, Dict[int, int]]):
    """Vista combinada: Top-10 de cada consulta con su relevancia según qrels.

    Los documentos que no están en qrels.tsv tienen relevancia 0.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "rank", "document_id", "score", "relevance", "relevant"])
        for qid in eval_qids:
            for rank, (doc_id, score) in enumerate(rankings[qid], start=1):
                rel = qrels[qid].get(doc_id, 0)
                writer.writerow([qid, rank, doc_id, f"{score:.4f}", rel, "Yes" if rel > 0 else "No"])


def run_benchmark(
    queries_path: Path,
    qrels_path: Path,
    rankings_output: Path = None,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
):
    """Ejecuta el benchmark completo sobre todas las configuraciones y genera reportes."""
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    missing = [qid for qid in queries if qid not in qrels]
    if missing:
        print(f"[AVISO] Consultas sin qrels (no se evalúan): {', '.join(missing)}")

    # Filtrar queries que tienen juicios en qrels
    eval_qids = [qid for qid in queries if qid in qrels]
    eval_qids.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    if not eval_qids:
        print("[ERROR] No hay consultas coincidentes entre queries.tsv y qrels.tsv.")
        return

    print("=" * 80)
    print("BENCHMARK DE EVALUACIÓN - LITERATURE RETRIEVAL")
    print(f"Consultas evaluadas: {len(eval_qids)} ({', '.join(eval_qids)})")
    print("Total documentos en corpus: 500 (Obras: Kundera, Homero, Gaarder, Hitler, Orwell)")
    print("=" * 80)

    all_results = {}          # cfg_key -> qid -> métricas
    rankings_by_config = {}   # cfg_key -> qid -> [(doc_id, score)]
    empty_queries = {}        # cfg_key -> [qid] con vector vacío

    for repr_name, feat, _, _ in CONFIGS:
        model = load_model(repr_name, feat, out_dir=models_dir)
        vectorizer = model["vectorizer"]
        matrix = model["matrix"]
        doc_ids = np.array(model["fragment_ids"])

        cfg_key = f"{repr_name}_{feat}"
        all_results[cfg_key] = {}
        rankings_by_config[cfg_key] = {}
        empty_queries[cfg_key] = []

        for qid in eval_qids:
            norm_q = normalize_query(queries[qid]["query"])
            q_vec = vectorizer.transform([norm_q])
            if q_vec.nnz == 0:
                empty_queries[cfg_key].append(qid)

            top_ranked = rank_documents(q_vec, matrix, doc_ids)
            all_results[cfg_key][qid] = evaluate_ranking(top_ranked, qrels[qid])
            rankings_by_config[cfg_key][qid] = top_ranked

    # Promedios por configuración y por nivel
    config_avgs = {cfg: average(list(res.values())) for cfg, res in all_results.items()}
    level_avgs = {}
    for cfg, res in all_results.items():
        for lvl in LEVEL_NAMES:
            qids_lvl = [q for q in eval_qids if queries[q]["level"] == lvl]
            if qids_lvl:
                level_avgs[(cfg, lvl)] = average([res[q] for q in qids_lvl])

    # --- TABLA 9: COMPARACIÓN DE CONFIGURACIONES ---
    print("\n" + "=" * 80)
    print("TABLA 9. Comparación de configuraciones (Promedios sobre las consultas)")
    print("=" * 80)
    print(f"{'Contenido':<10} {'Repr.':<12} {'N-gramas':<10} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'MRR':<8} {'nDCG@10':<8}")
    print("-" * 80)
    for repr_name, feat, label_repr, label_feat in CONFIGS:
        a = config_avgs[f"{repr_name}_{feat}"]
        print(f"{'Texto':<10} {label_repr:<12} {label_feat:<10} {a['p10']:<8.2f} {a['r10']:<8.2f} "
              f"{a['f1']:<8.2f} {a['rr']:<8.2f} {a['ndcg10']:<8.3f}")

    # Mejor configuración: mayor nDCG@10; en empate, mayor F1@10
    best_config = max(config_avgs, key=lambda k: (config_avgs[k]["ndcg10"], config_avgs[k]["f1"]))
    print("-" * 80)
    print(f">> MEJOR CONFIGURACIÓN DETECTADA: {best_config} <<")

    print("\nConsultas con vector vacío (sin términos en común con el vocabulario):")
    for repr_name, feat, label_repr, label_feat in CONFIGS:
        cfg_key = f"{repr_name}_{feat}"
        vacias = empty_queries[cfg_key]
        falsos = sum(1 for q in eval_qids for d, s in rankings_by_config[cfg_key][q]
                     if s <= 0 and d in qrels[q])
        print(f"  {label_repr:<11} {label_feat:<4} {len(vacias):>2}/{len(eval_qids)}  "
              f"{', '.join(vacias) or '-':<40} aciertos con score 0: {falsos}")

    # --- Archivos de salida ---
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ranking de cada configuración
    for cfg_key in rankings_by_config:
        write_rankings(results_dir / f"rankings_{cfg_key}.tsv", eval_qids, rankings_by_config[cfg_key])

    # Métricas por consulta y configuración
    with open(results_dir / "metrics_by_query.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["config", "representation", "feature", "query_id", "difficulty_level",
                         "p10", "r10", "f1_10", "rr", "ndcg10",
                         "empty_query_vector", "n_zero_score", "zero_score_hits"])
        for repr_name, feat, _, _ in CONFIGS:
            cfg_key = f"{repr_name}_{feat}"
            for qid in eval_qids:
                m = all_results[cfg_key][qid]
                ranking = rankings_by_config[cfg_key][qid]
                zero_docs = [d for d, s in ranking if s <= 0]
                writer.writerow([cfg_key, repr_name, feat, qid, queries[qid]["level"],
                                 *(f"{m[k]:.4f}" for k in METRICS),
                                 int(qid in empty_queries[cfg_key]),
                                 len(zero_docs),
                                 sum(1 for d in zero_docs if d in qrels[qid])])

    # Promedios por configuración (Tabla 9)
    with open(results_dir / "metrics_by_config.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["config", "representation", "feature", "p10", "r10", "f1_10", "mrr", "ndcg10",
                         "n_queries", "n_empty_query_vectors", "best"])
        for repr_name, feat, _, _ in CONFIGS:
            cfg_key = f"{repr_name}_{feat}"
            a = config_avgs[cfg_key]
            writer.writerow([cfg_key, repr_name, feat, *(f"{a[k]:.4f}" for k in METRICS),
                             len(eval_qids), len(empty_queries[cfg_key]), int(cfg_key == best_config)])

    # Promedios por nivel y configuración (Tabla 11 para las 6 configuraciones)
    with open(results_dir / "metrics_by_level.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["config", "difficulty_level", "level_name", "p10", "r10", "f1_10", "mrr", "ndcg10", "n_queries"])
        for repr_name, feat, _, _ in CONFIGS:
            cfg_key = f"{repr_name}_{feat}"
            for lvl, name in LEVEL_NAMES.items():
                if (cfg_key, lvl) in level_avgs:
                    a = level_avgs[(cfg_key, lvl)]
                    n_lvl = sum(1 for q in eval_qids if queries[q]["level"] == lvl)
                    writer.writerow([cfg_key, lvl, name, *(f"{a[k]:.4f}" for k in METRICS), n_lvl])

    # Vista combinada (ranking + relevancia) de cada configuración
    for repr_name, feat, _, _ in CONFIGS:
        cfg_key = f"{repr_name}_{feat}"
        write_combined_view(results_dir / f"combined_view_{cfg_key}.tsv", eval_qids,
                            rankings_by_config[cfg_key], qrels)

    print(f"\n>> Análisis de las 6 configuraciones guardado en {results_dir}/")

    # Ranking oficial: mejor configuración
    if rankings_output:
        write_rankings(Path(rankings_output), eval_qids, rankings_by_config[best_config])
        print(f">> Ranking oficial ({best_config}) guardado en {rankings_output}")
        combined_path = Path(rankings_output).with_name("combined_view.tsv")
        write_combined_view(combined_path, eval_qids, rankings_by_config[best_config], qrels)
        print(f">> Vista combinada oficial ({best_config}) guardada en {combined_path}")

    # --- TABLA 10: RESULTADOS POR CONSULTA (MEJOR CONFIGURACIÓN) ---
    print("\n" + "=" * 80)
    print(f"TABLA 10. Resultados por consulta (Configuración: {best_config})")
    print("=" * 80)
    print(f"{'query_id':<10} {'Nivel':<10} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'RR':<8} {'nDCG@10':<8}")
    print("-" * 80)

    best_res = all_results[best_config]
    for qid in eval_qids:
        lvl_str = LEVEL_NAMES.get(queries[qid]["level"], str(queries[qid]["level"]))
        m = best_res[qid]
        print(f"{qid:<10} {lvl_str:<10} {m['p10']:<8.2f} {m['r10']:<8.2f} {m['f1']:<8.2f} {m['rr']:<8.2f} {m['ndcg10']:<8.3f}")

    tot = config_avgs[best_config]
    print("-" * 80)
    print(f"{'Promedio':<21} {tot['p10']:<8.2f} {tot['r10']:<8.2f} {tot['f1']:<8.2f} {tot['rr']:<8.2f} {tot['ndcg10']:<8.3f}")

    # --- TABLA 11: PROMEDIOS POR NIVEL DE DIFICULTAD (MEJOR CONFIGURACIÓN) ---
    print("\n" + "=" * 80)
    print(f"TABLA 11. Promedios por nivel de dificultad (Configuración: {best_config})")
    print("=" * 80)
    print(f"{'Nivel':<12} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'MRR':<8} {'nDCG@10':<8}")
    print("-" * 80)
    for lvl, name in LEVEL_NAMES.items():
        if (best_config, lvl) in level_avgs:
            a = level_avgs[(best_config, lvl)]
            print(f"{name:<12} {a['p10']:<8.2f} {a['r10']:<8.2f} {a['f1']:<8.2f} {a['rr']:<8.2f} {a['ndcg10']:<8.3f}")
    print("-" * 80)
    print(f"{'Global':<12} {tot['p10']:<8.2f} {tot['r10']:<8.2f} {tot['f1']:<8.2f} {tot['rr']:<8.2f} {tot['ndcg10']:<8.3f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el benchmark sobre el corpus")
    parser.add_argument("--queries", default=str(ROOT / "queries.tsv"), help="Ruta a queries.tsv")
    parser.add_argument("--qrels", default=str(ROOT / "qrels.tsv"), help="Ruta a qrels.tsv")
    parser.add_argument("--output", default=str(ROOT / "rankings.tsv"), help="Ruta para rankings.tsv")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Carpeta para rankings y métricas de las 6 configuraciones")
    args = parser.parse_args()

    run_benchmark(
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        rankings_output=Path(args.output),
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    main()