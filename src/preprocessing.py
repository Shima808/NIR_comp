# -*- coding: utf-8 -*-
"""
preprocessing.py
================
NIRスペクトルの前処理モジュール。

現ベスト構成（LB 17.200）の前処理を担当する:
    SNV（全1555波長） → 5500 cm⁻¹以上の帯域を選択

重要な制約（calibration transfer 問題）:
    - train(13樹種) と test(6樹種) は完全に別種
    - `species number`・`樹種` は特徴量に絶対に使わない
    - そのためメタ列を明示的に除外してスペクトル列のみを取り出す

物理的根拠:
    5200 cm⁻¹付近（O-Hの伸縮+変角の結合音）が高含水率で飽和するため、
    5500 cm⁻¹以上のみを使うと種間転移性が向上する（北海道大 O-7論文）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルート（このファイルの2つ上 = NIR.comp/）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# 特徴量に使ってはいけないメタ列（cp932の日本語列名）
META_COLS = ["sample number", "species number", "樹種", "含水率"]

# 物理知見に基づく固定カットオフ（CV最適化はしない）
CUTOFF = 5500.0  # cm⁻¹

ENCODING = "cp932"


def load_train(data_dir: Path = DATA_DIR):
    """学習データを読み込む。

    Returns:
        X        : スペクトル行列 (n_samples, 1555)
        y        : 含水率 (n_samples,)
        groups   : 樹種ラベル（LOSO/GroupKFold用、特徴量ではない）
        spec_cols: スペクトル列名のリスト（波数文字列）
    """
    df = pd.read_csv(data_dir / "train_near.csv", encoding=ENCODING)
    spec_cols = [c for c in df.columns if c not in META_COLS]
    X = df[spec_cols].values.astype(float)
    y = df["含水率"].values.astype(float)
    groups = df["樹種"].values
    return X, y, groups, spec_cols


def load_test(data_dir: Path = DATA_DIR):
    """テストデータを読み込む。

    Returns:
        X            : スペクトル行列 (n_samples, 1555)
        sample_number: 提出CSV用のサンプル番号
        spec_cols    : スペクトル列名のリスト（波数文字列）
    """
    df = pd.read_csv(data_dir / "test_near.csv", encoding=ENCODING)
    # testには含水率列が無いので META から除いて判定
    meta = [c for c in META_COLS if c in df.columns]
    spec_cols = [c for c in df.columns if c not in meta]
    X = df[spec_cols].values.astype(float)
    sample_number = df["sample number"].values
    return X, sample_number, spec_cols


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate。各サンプル（行）を平均0・分散1に正規化する。

    散乱（ベースライン変動・光路長差）を1サンプル内で補正する。
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / std


def band_mask_from_cols(spec_cols, cutoff: float = CUTOFF) -> np.ndarray:
    """スペクトル列名（波数）から cutoff 以上を選択する真偽マスクを作る。"""
    wavenums = np.array([float(c) for c in spec_cols])
    return wavenums >= cutoff


def preprocess(X: np.ndarray, spec_cols, cutoff: float = CUTOFF) -> np.ndarray:
    """現ベストの前処理パイプライン: SNV（全波長） → cutoff以上を選択。

    学習・予測の両方でこの関数を通すことで前処理の一貫性を保証する。
    """
    X_snv = snv(X)
    mask = band_mask_from_cols(spec_cols, cutoff)
    return X_snv[:, mask]


if __name__ == "__main__":
    # 動作確認用
    X, y, groups, spec_cols = load_train()
    Xp = preprocess(X, spec_cols)
    print(f"Train      : {X.shape}")
    print(f"前処理後   : {Xp.shape}  (cutoff {CUTOFF} cm⁻¹以上)")
    print(f"樹種数     : {len(set(groups))}")
    print(f"含水率範囲 : {y.min():.2f} - {y.max():.2f} %")
