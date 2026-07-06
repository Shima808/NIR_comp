# -*- coding: utf-8 -*-
"""
learning.py
===========
学習モジュール。現ベスト構成（LB 17.200）のモデルを学習し、保存する。

構成:
    SNV（全波長） → 5500 cm⁻¹以上 → PLS(n_components=3) → clip[y_min, y_max]

PLS3 の根拠:
    成分数を増やすと train RMSE は下がるが LOSO は悪化する（過学習）。
    n=3 は過学習の崖の直前にある最適点（成分数スキャンで実証済み）。

このスクリプトは:
    1. GroupKFold CV と LOSO で汎化性能を診断（参考値）
    2. 全trainでPLSを学習
    3. モデルと前処理パラメータを models/ に保存（predicting.pyが読む）
"""
from pathlib import Path

import numpy as np
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from preprocessing import ROOT, CUTOFF, load_train, preprocess

# ハイパーパラメータ（物理知見とCV診断で確定。探索的最適化はしない）
N_COMPONENTS = 3
N_SPLITS = 5

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pls3_5500up.joblib"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cv_diagnose(X, y, groups, n_components=N_COMPONENTS, n_splits=N_SPLITS):
    """GroupKFold CV（同一樹種をtrain/valに混ぜない）でOOF予測を作り、RMSEを返す。

    注意: GroupKFold CVはLBより約8〜9低めに出る（参考値）。
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[tr], y[tr])
        oof[va] = pls.predict(X[va]).ravel()
        print(f"  Fold {fold+1}  val樹種={set(groups[va])}  RMSE={rmse(y[va], oof[va]):.4f}")
    cv = rmse(y, oof)
    print(f"  GroupKFold CV RMSE : {cv:.4f}（LBより約8〜9低め）")
    return cv


def loso_diagnose(X, y, groups, n_components=N_COMPONENTS):
    """Leave-One-Species-Out。樹種ごとに抜き出して評価。

    LOSO median が LB と最も相関する指標（meanより信頼）。
    """
    scores = {}
    for sp in np.unique(groups):
        va = groups == sp
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[~va], y[~va])
        scores[sp] = rmse(y[va], pls.predict(X[va]).ravel())
    vals = np.array(list(scores.values()))
    for sp, s in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"    {sp:<12} {s:6.2f}")
    print(f"  LOSO mean   : {vals.mean():.4f}")
    print(f"  LOSO median : {np.median(vals):.4f}  ← LBと相関")
    return scores


def train_and_save(diagnose: bool = True):
    """現ベストモデルを学習し、前処理パラメータごと保存する。"""
    X_raw, y, groups, spec_cols = load_train()
    X = preprocess(X_raw, spec_cols, CUTOFF)
    y_min, y_max = float(y.min()), float(y.max())

    print(f"学習データ : {X_raw.shape} → 前処理後 {X.shape}")
    print(f"clip範囲   : [{y_min:.2f}, {y_max:.2f}]\n")

    if diagnose:
        print("[GroupKFold CV]")
        cv_diagnose(X, y, groups)
        print("\n[LOSO（樹種別）]")
        loso_diagnose(X, y, groups)
        print()

    # 全trainで最終学習
    model = PLSRegression(n_components=N_COMPONENTS)
    model.fit(X, y)

    # モデル＋前処理パラメータをバンドルして保存
    bundle = {
        "model": model,
        "cutoff": CUTOFF,
        "n_components": N_COMPONENTS,
        "y_min": y_min,
        "y_max": y_max,
        "spec_cols": spec_cols,  # 列順の整合性チェック用
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"モデル保存完了: {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    train_and_save(diagnose=True)
