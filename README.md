# NIR木材含水率予測コンペ（Signate）

近赤外（NIR）スペクトルから木材の含水率(%)を予測する回帰タスク（評価指標: RMSE）の取り組み記録。

## タスクの核心

- **入力**: NIRスペクトル 1555波長（3999.8〜9993.8 cm⁻¹, Bruker MATRIX-F）
- **出力**: 含水率(%)（train範囲 0.84〜298.58）
- **最重要制約**: Train(13樹種) と Test(6樹種) が**完全に別種** → **calibration transfer 問題**
  - `species number`・樹種列は特徴量に使わない（使うとリーク）

## 最終結果

| | Public | Private |
|---|---|---|
| 提出① `pls3_5500up` | 17.200 | 13.796 |
| 提出② `ens5400_5500_w42` | 17.216 | 13.786 |
| **選ばなかった `pls3_snv`（素の全波長SNV+PLS3）** | 19.592 | **9.780**（13〜15位相当） |

**最大の学び — Public LBへの過適応で汎化していたモデルを自分で捨てた。**
5500 cm⁻¹カットオフは Public を改善したが Private では悪化。詳細は [context/retrospective.md](context/retrospective.md)。

> **教訓: 構造の正しいCV（GroupKFold/LOSO）の素直な集約値を信じる。LBに合わせて指標を後付けしない。**

## ベストモデル（再現構成）

```
SNV(全波長) → 5500 cm⁻¹以上を選択 → PLS(n_components=3) → clip[0.84, 298.58]
```

物理的根拠: 5200 cm⁻¹付近（O-Hの結合音）が高含水率で飽和するため、飽和帯域を除去すると種間転移性が向上する（北海道大 O-7論文）。

## リポジトリ構成

```
├── src/                   提出用3分割モジュール
│   ├── preprocessing.py   SNV＋5500cm⁻¹カットオフ
│   ├── learning.py        PLS3学習＋CV/LOSO診断＋モデル保存
│   └── predicting.py      モデル読込＋予測＋提出CSV出力
├── notebooks/             EDA〜帯域カットオフ（01〜09）
├── submissions/           全提出CSV履歴
├── models/                学習済みモデル（pls3_5500up.joblib）
├── context/               引き継ぎ・実行環境・振り返り
└── CLAUDE.md              プロジェクト方針
```

> **注**: コンペ配布データ（`data/`）と論文PDF（`literature/`）は、Signate規約・著作権のため本リポジトリには含めていません。

## 実行方法

`data/train_near.csv`・`data/test_near.csv`（cp932）をローカルに配置した上で:

```bash
cd src
python learning.py     # 診断表示 + models/pls3_5500up.joblib を保存
python predicting.py   # モデル読込 → submissions/submission_pls3_5500up.csv 生成
```

依存: numpy / pandas / scikit-learn / scipy / joblib

## 確定した失敗パターン（再現禁止）

- 樹種特徴量 / random KFold → 完全リーク
- 非線形モデル（SVR / LightGBM）→ 訓練樹種に過剰適合
- EPO系（種平均 / 乾燥曲線）→ 水分信号まで除去
- **Public LBへの帯域チューニング（5500カットオフ）→ Privateで悪化**（今回の敗因）
