"""Evaluación del Benchmark y Generación de Métricas - Proyecto Literature Retrieval.

Calcula P@10, R@10, F1@10, MRR y nDCG@10 sobre las 6 configuraciones VSM
(Frecuencia, Binaria y TF-IDF x Unigramas y Bigramas) sobre todo el corpus (500 documentos).
Genera el archivo oficial rankings.tsv.

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
from .vsm import FEATURES, REPRESENTATIONS, load_model

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


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
    """Calcula P@10, R@10, F1@10, RR y nDCG@10 para un Top-10 de documentos."""
    total_rel_in_benchmark = len(qrel)
    if total_rel_in_benchmark == 0:
        return {"p10": 0.0, "r10": 0.0, "f1": 0.0, "rr": 0.0, "ndcg10": 0.0}

    # P@10 y R@10 (cualquier relevancia > 0 cuenta como relevante)
    rel_retrieved = sum(1 for doc_id, _ in top_docs if doc_id in qrel)
    p10 = rel_retrieved / 10.0
    r10 = rel_retrieved / float(total_rel_in_benchmark)
    f1 = (2.0 * p10 * r10 / (p10 + r10)) if (p10 + r10) > 0 else 0.0

    # Reciprocal Rank (RR): 1 / rank del primer relevante
    rr = 0.0
    for rank, (doc_id, _) in enumerate(top_docs, start=1):
        if doc_id in qrel:
            rr = 1.0 / rank
            break

    # nDCG@10: Ganancia graduada (2^rel - 1) con descuento log2(rank + 1)
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(top_docs, start=1):
        rel = qrel.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(qrel.values(), reverse=True)[:10]
    idcg = sum((2**r - 1) / math.log2(rank + 1) for rank, r in enumerate(ideal_rels, start=1))
    ndcg10 = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "p10": p10,
        "r10": r10,
        "f1": f1,
        "rr": rr,
        "ndcg10": ndcg10,
    }


def run_benchmark(
    queries_path: Path,
    qrels_path: Path,
    rankings_output: Path = None,
    models_dir: Path = MODELS_DIR,
):
    """Ejecuta el benchmark completo sobre todas las configuraciones y genera reportes."""
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    # Filtrar queries que tienen juicios en qrels
    eval_qids = [qid for qid in queries if qid in qrels]
    eval_qids.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    if not eval_qids:
        print("[ERROR] No hay consultas coincidentes entre queries.tsv y qrels.tsv.")
        return

    print("=" * 80)
    print(f"BENCHMARK DE EVALUACIÓN - LITERATURE RETRIEVAL")
    print(f"Consultas evaluadas: {len(eval_qids)} ({', '.join(eval_qids)})")
    print(f"Total documentos en corpus: 500 (Obras: Kundera, Homero, Gaarder, Hitler, Orwell)")
    print("=" * 80)

    # Configuraciones oficiales (Tabla 9)
    configs = [
        ("frequency", "unigram", "Frecuencia", "Uni"),
        ("frequency", "bigram", "Frecuencia", "Bi"),
        ("binary", "unigram", "Binaria", "Uni"),
        ("binary", "bigram", "Binaria", "Bi"),
        ("tfidf", "unigram", "TF-IDF", "Uni"),
        ("tfidf", "bigram", "TF-IDF", "Bi"),
    ]

    all_results = {}
    rankings_by_config = {}

    for repr_name, feat, label_repr, label_feat in configs:
        model = load_model(repr_name, feat, out_dir=models_dir)
        vectorizer = model["vectorizer"]
        matrix = model["matrix"]
        doc_ids = np.array(model["fragment_ids"])

        cfg_key = f"{repr_name}_{feat}"
        all_results[cfg_key] = {}
        rankings_by_config[cfg_key] = {}

        for qid in eval_qids:
            q_info = queries[qid]
            norm_q = normalize_query(q_info["query"])
            q_vec = vectorizer.transform([norm_q])
            sims = cosine_similarity(q_vec, matrix)[0]

            top_indices = np.argsort(sims)[::-1][:10]
            top_ranked = [(int(doc_ids[idx]), float(sims[idx])) for idx in top_indices]

            metrics = evaluate_ranking(top_ranked, qrels[qid])
            all_results[cfg_key][qid] = metrics
            rankings_by_config[cfg_key][qid] = top_ranked

    # --- TABLA 9: COMPARACIÓN DE CONFIGURACIONES ---
    print("\n" + "=" * 80)
    print("TABLA 9. Comparación de configuraciones (Promedios sobre las consultas)")
    print("=" * 80)
    print(f"{'Contenido':<10} {'Repr.':<12} {'N-gramas':<10} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'MRR':<8} {'nDCG@10':<8}")
    print("-" * 80)

    config_averages = {}
    for repr_name, feat, label_repr, label_feat in configs:
        cfg_key = f"{repr_name}_{feat}"
        res = all_results[cfg_key]
        n = len(res)
        avg_p10 = sum(m["p10"] for m in res.values()) / n
        avg_r10 = sum(m["r10"] for m in res.values()) / n
        avg_f1 = sum(m["f1"] for m in res.values()) / n
        avg_mrr = sum(m["rr"] for m in res.values()) / n
        avg_ndcg = sum(m["ndcg10"] for m in res.values()) / n

        config_averages[cfg_key] = (avg_ndcg, avg_f1)
        print(f"{'Texto':<10} {label_repr:<12} {label_feat:<10} {avg_p10:<8.2f} {avg_r10:<8.2f} {avg_f1:<8.2f} {avg_mrr:<8.2f} {avg_ndcg:<8.3f}")

    # Determinar la mejor configuración (mayor nDCG y F1)
    best_config = max(config_averages, key=lambda k: config_averages[k])
    print("-" * 80)
    print(f">> MEJOR CONFIGURACIÓN DETECTADA: {best_config} <<")

    # Guardar archivo oficial rankings.tsv con la mejor configuración
    if rankings_output:
        with open(rankings_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["query_id", "document_id", "rank", "score"])
            for qid in eval_qids:
                for rank, (doc_id, score) in enumerate(rankings_by_config[best_config][qid], start=1):
                    writer.writerow([qid, doc_id, rank, f"{score:.4f}"])
        print(f">> Predicciones guardadas en {rankings_output}")

    # --- TABLA 10: RESULTADOS POR CONSULTA (MEJOR CONFIGURACIÓN) ---
    print("\n" + "=" * 80)
    print(f"TABLA 10. Resultados por consulta (Configuración: {best_config})")
    print("=" * 80)
    print(f"{'query_id':<10} {'Nivel':<10} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'RR':<8} {'nDCG@10':<8}")
    print("-" * 80)

    best_res = all_results[best_config]
    level_names = {1: "Fácil", 2: "Media", 3: "Difícil"}

    for qid in eval_qids:
        lvl_str = level_names.get(queries[qid]["level"], str(queries[qid]["level"]))
        m = best_res[qid]
        print(f"{qid:<10} {lvl_str:<10} {m['p10']:<8.2f} {m['r10']:<8.2f} {m['f1']:<8.2f} {m['rr']:<8.2f} {m['ndcg10']:<8.3f}")

    n_tot = len(best_res)
    tot_p10 = sum(m["p10"] for m in best_res.values()) / n_tot
    tot_r10 = sum(m["r10"] for m in best_res.values()) / n_tot
    tot_f1 = sum(m["f1"] for m in best_res.values()) / n_tot
    tot_mrr = sum(m["rr"] for m in best_res.values()) / n_tot
    tot_ndcg = sum(m["ndcg10"] for m in best_res.values()) / n_tot
    print("-" * 80)
    print(f"{'Promedio':<21} {tot_p10:<8.2f} {tot_r10:<8.2f} {tot_f1:<8.2f} {tot_mrr:<8.2f} {tot_ndcg:<8.3f}")

    # --- TABLA 11: PROMEDIOS POR NIVEL DE DIFICULTAD ---
    print("\n" + "=" * 80)
    print("TABLA 11. Promedios por nivel de dificultad")
    print("=" * 80)
    print(f"{'Nivel':<12} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'MRR':<8} {'nDCG@10':<8}")
    print("-" * 80)

    for lvl, name in [(1, "Fácil"), (2, "Media"), (3, "Difícil")]:
        qids_lvl = [qid for qid in eval_qids if queries[qid]["level"] == lvl]
        if qids_lvl:
            lp10 = sum(best_res[q]["p10"] for q in qids_lvl) / len(qids_lvl)
            lr10 = sum(best_res[q]["r10"] for q in qids_lvl) / len(qids_lvl)
            lf1 = sum(best_res[q]["f1"] for q in qids_lvl) / len(qids_lvl)
            lmrr = sum(best_res[q]["rr"] for q in qids_lvl) / len(qids_lvl)
            lndcg = sum(best_res[q]["ndcg10"] for q in qids_lvl) / len(qids_lvl)
            print(f"{name:<12} {lp10:<8.2f} {lr10:<8.2f} {lf1:<8.2f} {lmrr:<8.2f} {lndcg:<8.3f}")

    print("-" * 80)
    print(f"{'Global':<12} {tot_p10:<8.2f} {tot_r10:<8.2f} {tot_f1:<8.2f} {tot_mrr:<8.2f} {tot_ndcg:<8.3f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el benchmark sobre el corpus")
    parser.add_argument("--queries", default=str(ROOT / "queries.tsv"), help="Ruta a queries.tsv")
    parser.add_argument("--qrels", default=str(ROOT / "qrels.tsv"), help="Ruta a qrels.tsv")
    parser.add_argument("--output", default=str(ROOT / "rankings.tsv"), help="Ruta para rankings.tsv")
    args = parser.parse_args()

    run_benchmark(
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        rankings_output=Path(args.output),
    )


if __name__ == "__main__":
    main()
