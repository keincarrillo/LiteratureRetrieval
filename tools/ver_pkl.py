"""Exporta las 6 representaciones VSM a CSV para poder verlas completas.
Uso:
    python ver_pkl.py models              -> formato largo: fragment_id, termino, valor (sin ceros)
    python ver_pkl.py models --completo   -> matriz COMPLETA con ceros, girada: filas = terminos, columnas = documentos
"""
import csv
import sys
from pathlib import Path

import joblib

args = [a for a in sys.argv[1:] if not a.startswith("--")]
completo = "--completo" in sys.argv
carpeta = Path(args[0] if args else "models")
salida = carpeta / ("csv_completo" if completo else "csv")
salida.mkdir(exist_ok=True)

print(f"{'representacion':20s} {'docs':>5s} {'terminos':>9s} {'no-ceros':>9s} {'% ceros':>8s}")
for pkl in sorted(carpeta.glob("*.pkl")):
    data = joblib.load(pkl)
    X = data["matrix"].tocoo()
    terminos = data["vectorizer"].get_feature_names_out()
    ids = data["fragment_ids"]

    if completo:
        XT = data["matrix"].T.tocsr()          # terminos x documentos
        with open(salida / f"{pkl.stem}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["termino"] + ids)
            for i, termino in enumerate(terminos):
                fila = XT[i].toarray()[0]
                w.writerow([termino] + [f"{v:.4g}" for v in fila])
    else:
        with open(salida / f"{pkl.stem}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fragment_id", "termino", "valor"])
            for fila, col, val in sorted(zip(X.row, X.col, X.data)):
                w.writerow([ids[fila], terminos[col], round(float(val), 4)])

    total = X.shape[0] * X.shape[1]
    print(f"{pkl.stem:20s} {X.shape[0]:5d} {X.shape[1]:9d} {X.nnz:9d} {100 * (1 - X.nnz / total):7.2f}%")

print(f"\nCSV guardados en {salida}/")