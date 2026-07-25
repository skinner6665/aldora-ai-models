"""
Treinamento XGBoost Sepse — PhysioNet Challenge 2019
Uso: python train_sepse.py --data_dir /path/to/physionet2019/training

Dataset: https://physionet.org/content/challenge-2019/1.0.0/
Formato PSV — colunas conforme FEATURE_NAMES em models/sepse.py
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURE_NAMES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
]
TARGET = "SepsisLabel"


def carregar_psv(data_dir: str) -> pd.DataFrame:
    arquivos = glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
    if not arquivos:
        arquivos = glob.glob(os.path.join(data_dir, "*.psv"))
    print(f"Encontrados {len(arquivos)} arquivos PSV.")

    dfs = []
    for path in arquivos:
        df = pd.read_csv(path, sep="|")
        df["HoraNoUTI"] = np.arange(len(df), dtype=float)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Diretório com arquivos .psv do PhysioNet 2019")
    parser.add_argument("--output", default="checkpoints/sepse_xgb.json", help="Caminho para salvar o modelo")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    print("Carregando dados...")
    df = carregar_psv(args.data_dir)

    features_presentes = [c for c in FEATURE_NAMES if c in df.columns]
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"Colunas ausentes (imputadas com 0): {missing}")
        for c in missing:
            df[c] = 0.0

    X = df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
    y = df[TARGET].fillna(0).values.astype(int)

    print(f"Total amostras: {len(X)}, Positivos: {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos:.2f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos,
        "seed": 42,
        "nthread": -1,
    }

    print("Treinando XGBoost...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    preds = booster.predict(dval)
    auc = roc_auc_score(y_val, preds)
    print(f"\nAUC validação: {auc:.4f}")

    booster.save_model(args.output)
    print(f"Modelo salvo em: {args.output}")
    print("\nFeature importance (top 10):")
    fi = booster.get_score(importance_type="gain")
    for nome, score in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        print(f"  {nome}: {score:.2f}")


if __name__ == "__main__":
    main()
