"""
Actividad Final - Ecosistema analítico End-to-End (scouting deportivo)

Entrena, evalúa y serializa el modelo de scouting:
  1. Ingesta desde MySQL e imputación (CRISP-DM: datos / preparación).
  2. Diagnóstico de overfitting: árbol sin control vs árbol con poda previa.
  3. Optimización con GridSearchCV: árbol único y Random Forest.
  4. Evaluación: ROC-AUC e importancia de variables (Gini).
  5. Exportación de models/modelo_scouting_rf.joblib y de las figuras del informe.

Uso (desde la carpeta que contiene src/):
    python exportar_modelo.py
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.database.connection2 import MySQLConnector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figuras")
RUTA_MODELO = os.path.join(MODELS_DIR, "modelo_scouting_rf.joblib")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 65)
    print("  ACTIVIDAD FINAL - ENTRENAMIENTO Y EXPORTACIÓN DEL MODELO")
    print("=" * 65 + "\n")

    # -------------------------------------------------------------
    # FASE 1. INGESTA Y PREPROCESAMIENTO
    # -------------------------------------------------------------
    db = MySQLConnector()
    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """
    print("[FASE 1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)
    if df.empty:
        print("[ERROR] No se obtuvieron datos de la base de datos.")
        return

    df["estatura_cm"] = pd.to_numeric(df["estatura_cm"], errors="coerce")
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["valor_mercado"] = pd.to_numeric(df["valor_mercado"], errors="coerce")

    # Imputación por mediana agrupada por posición
    df["posicion"] = df["posicion"].fillna("Sin Posicion")
    df["estatura_cm"] = df.groupby("posicion")["estatura_cm"].transform(lambda x: x.fillna(x.median()))
    df["estatura_cm"] = df["estatura_cm"].fillna(df["estatura_cm"].median())
    df["edad"] = df["edad"].fillna(df["edad"].median())

    # Target binario por percentil 75 (Top 25 % del valor de mercado)
    umbral_elite = float(df["valor_mercado"].quantile(0.75))
    df["es_elite"] = (df["valor_mercado"] >= umbral_elite).astype(int)

    # One-Hot Encoding
    df_dummies = pd.get_dummies(df[["posicion"]], drop_first=True, dtype=int)
    X = pd.concat([df[["edad", "estatura_cm"]], df_dummies], axis=1)
    y = df["es_elite"]

    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]
        df = df[~filas_con_nan]

    # Muestreo estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Registros: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[OK] Umbral Élite (percentil 75): {umbral_elite:,.0f} EUR")
    print(f"[OK] Estándar: {(y == 0).sum()} | Élite: {(y == 1).sum()} ({y.mean()*100:.1f}% Élite)\n")

    # Apoyo descriptivo para interpretar la importancia de variables
    desc = df.assign(es_elite=y)
    elite_pos = desc.groupby("posicion")["es_elite"].mean().round(4).to_dict()
    estatura_pos = desc.groupby("posicion")["estatura_cm"].mean().round(1).to_dict()
    tramos = pd.cut(desc["edad"], [0, 19, 23, 27, 31, 100],
                    labels=["<=19", "20-23", "24-27", "28-31", "32+"])
    elite_edad = desc.groupby(tramos, observed=True)["es_elite"].mean().round(4).to_dict()
    elite_edad = {str(k): v for k, v in elite_edad.items()}

    # -------------------------------------------------------------
    # FASE 2. DIAGNÓSTICO DE OVERFITTING
    # -------------------------------------------------------------
    print("[FASE 2] Árbol sin control vs árbol con poda previa...")
    tree_libre = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_train, y_train)
    tree_podado = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42).fit(X_train, y_train)

    def evaluar(nombre, modelo):
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        return {
            "modelo": nombre,
            "acc_train": accuracy_score(y_train, modelo.predict(X_train)),
            "acc_test": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    res = {
        "arbol_libre": evaluar("Árbol sin control", tree_libre),
        "arbol_podado": evaluar("Árbol con poda previa", tree_podado),
    }
    for k in ("arbol_libre", "arbol_podado"):
        r = res[k]
        print(f"  • {r['modelo']}: Train {r['acc_train']:.4f} | Test {r['acc_test']:.4f} "
              f"| Brecha {(r['acc_train']-r['acc_test'])*100:.2f}% | F1 Élite {r['f1']:.4f}")
    print()

    # -------------------------------------------------------------
    # FASE 3. OPTIMIZACIÓN: ÁRBOL ÚNICO Y RANDOM FOREST
    # -------------------------------------------------------------
    print("[FASE 3a] GridSearchCV - Árbol único optimizado...")
    grid_arbol = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [2, 3, 4, 5, 6],
            "min_samples_leaf": [3, 5, 10, 15],
            "criterion": ["gini", "entropy"],
            "class_weight": ["balanced", None],
        },
        cv=5, scoring="f1", n_jobs=-1,
    ).fit(X_train, y_train)
    tree_opt = grid_arbol.best_estimator_
    res["arbol_optimo"] = evaluar("Árbol único optimizado", tree_opt)
    print(f"  • Mejores hiperparámetros: {grid_arbol.best_params_}")
    print(f"  • F1 test: {res['arbol_optimo']['f1']:.4f} | AUC: {res['arbol_optimo']['auc']:.4f}\n")

    print("[FASE 3b] GridSearchCV - Random Forest (puede tardar unos minutos)...")
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, 14],
        "min_samples_leaf": [5, 10, 20],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid_rf, cv=5, scoring="f1", n_jobs=-1,
    ).fit(X_train, y_train)
    rf = grid_rf.best_estimator_
    res["random_forest"] = evaluar("Random Forest optimizado", rf)
    print(f"  • Mejores hiperparámetros: {grid_rf.best_params_}")
    print(f"  • F1 medio en validación cruzada: {grid_rf.best_score_:.4f}")
    r = res["random_forest"]
    print(f"  • Test -> Accuracy {r['acc_test']:.4f} | Precisión {r['precision']:.4f} | "
          f"Recall {r['recall']:.4f} | F1 {r['f1']:.4f} | AUC {r['auc']:.4f}")
    print(f"  • Matriz: TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}\n")

    # -------------------------------------------------------------
    # FASE 4. EVALUACIÓN: FIGURAS
    # -------------------------------------------------------------
    print("[FASE 4] Generando figuras del informe...")

    # 4.1 Overfitting: Train vs Test
    fig, ax = plt.subplots(figsize=(7, 4.6))
    etapas = ["Árbol sin restricciones", "Árbol con poda previa"]
    tr = [res["arbol_libre"]["acc_train"], res["arbol_podado"]["acc_train"]]
    te = [res["arbol_libre"]["acc_test"], res["arbol_podado"]["acc_test"]]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w/2, tr, w, label="Train", color="#c0392b", edgecolor="black")
    b2 = ax.bar(x + w/2, te, w, label="Test", color="#2e86c1", edgecolor="black")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.4f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(etapas)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Diagnóstico de overfitting: Accuracy en Train vs Test", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_overfitting.png"), dpi=200)
    plt.close()

    # 4.2 Curvas ROC comparativas
    fig, ax = plt.subplots(figsize=(7, 5.6))
    curvas = [
        ("Árbol sin control", tree_libre, "#7f8c8d", "--"),
        ("Árbol único optimizado", tree_opt, "#e67e22", "-"),
        ("Random Forest optimizado", rf, "#1e8449", "-"),
    ]
    for nombre, modelo, color, ls in curvas:
        proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2.2, label=f"{nombre} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1.2, label="Azar (AUC = 0.500)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax.set_title("Curvas ROC: Árbol único vs Random Forest", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_roc.png"), dpi=200)
    plt.close()

    # 4.3 Importancia de variables (Gini)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.barh(imp.index, imp.values, color="#2874a6", edgecolor="black")
    for b in bars:
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2, f"{b.get_width()*100:.1f}%",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, imp.max() * 1.15)
    ax.set_xlabel("Importancia de Gini (suma = 1)")
    ax.set_title("Importancia de variables - Random Forest", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_importancia.png"), dpi=200)
    plt.close()
    print("[OK] Figuras guardadas en la carpeta 'figuras/'.\n")

    # -------------------------------------------------------------
    # FASE 5. SERIALIZACIÓN (JOBLIB)
    # -------------------------------------------------------------
    paquete = {
        "modelo": rf,
        "columnas": list(X.columns),
        "posiciones": sorted(df_dummies.columns.str.replace("posicion_", "", regex=False).tolist() + ["Attack"]),
        "umbral_elite_eur": umbral_elite,
        "hiperparametros": grid_rf.best_params_,
        "metricas_test": {k: res["random_forest"][k] for k in ("acc_test", "precision", "recall", "f1", "auc")},
        "version_sklearn": sklearn.__version__,
    }
    joblib.dump(paquete, RUTA_MODELO, compress=3)
    print(f"[FASE 5] Modelo exportado en: {RUTA_MODELO}")
    print(f"         Tamaño: {os.path.getsize(RUTA_MODELO)/1024:.0f} KB | scikit-learn {sklearn.__version__}\n")

    # Resumen reproducible (lo usa el informe)
    resumen = {
        "fecha_filas": int(len(X)),
        "train": int(len(X_train)), "test": int(len(X_test)),
        "umbral_elite_eur": umbral_elite,
        "estandar": int((y == 0).sum()), "elite": int((y == 1).sum()),
        "baseline_test": float((y_test == 0).mean()),
        "resultados": res,
        "mejores_params_arbol": grid_arbol.best_params_,
        "mejores_params_rf": grid_rf.best_params_,
        "f1_cv_rf": float(grid_rf.best_score_),
        "importancias": {k: float(v) for k, v in imp.sort_values(ascending=False).items()},
        "elite_por_posicion": elite_pos,
        "estatura_media_por_posicion": estatura_pos,
        "elite_por_edad": elite_edad,
    }
    with open(os.path.join(FIG_DIR, "resumen_resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("[RESUMEN] Importancia de variables (Gini):")
    for k, v in imp.sort_values(ascending=False).items():
        print(f"  • {k:<22}{v*100:6.2f}%")
    print("\n[OK] Proceso finalizado.")


if __name__ == "__main__":
    main()
