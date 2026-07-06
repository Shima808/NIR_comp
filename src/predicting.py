# -*- coding: utf-8 -*-
"""
predicting.py
=============
予測モジュール。learning.py が保存したモデルを読み込み、
testスペクトルから含水率を予測して提出CSVを出力する。

提出フォーマット:
    ヘッダーなし・2列（sample number, 含水率）

clip:
    予測値を学習データの含水率レンジ [y_min, y_max] に丸める
    （NIRの外挿は信頼できないため）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from preprocessing import ROOT, load_test, preprocess
from learning import MODEL_PATH

SUBMIT_DIR = ROOT / "submissions"
SUBMIT_PATH = SUBMIT_DIR / "submission_pls3_5500up.csv"


def load_bundle(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {model_path}\n"
            "先に learning.py を実行してモデルを学習・保存してください。"
        )
    return joblib.load(model_path)


def predict(bundle, X_raw, spec_cols):
    """前処理 → PLS予測 → clip。学習時と同じ前処理関数を通す。"""
    # 列順の整合性チェック（train/testで波長並びが一致していること）
    if list(spec_cols) != list(bundle["spec_cols"]):
        raise ValueError("testのスペクトル列が学習時と一致しません。")

    X = preprocess(X_raw, spec_cols, bundle["cutoff"])
    y_pred = bundle["model"].predict(X).ravel()
    return np.clip(y_pred, bundle["y_min"], bundle["y_max"])


def main():
    bundle = load_bundle()
    X_raw, sample_number, spec_cols = load_test()
    y_pred = predict(bundle, X_raw, spec_cols)

    print(f"予測完了: n={len(y_pred)}  min={y_pred.min():.2f}  max={y_pred.max():.2f}")

    SUBMIT_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"sample number": sample_number, "含水率": y_pred})
    submission.to_csv(SUBMIT_PATH, index=False, header=False, encoding="cp932")
    print(f"提出CSV保存完了: {SUBMIT_PATH}  (LB 17.200 を再現)")


if __name__ == "__main__":
    main()
