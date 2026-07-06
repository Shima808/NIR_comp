---
title: 実行環境・提出物チェックリスト
updated: 2026-06-22
---

# 実行環境

## プラットフォーム

| 項目 | 内容 |
|------|------|
| 実行環境 | Google Colaboratory（無料枠） |
| OS | Linux（Colab 標準） |
| Python | 3.10.x（Colab デフォルト） |
| ハードウェア | CPU ランタイム（GPU 不使用） |

## 使用ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| numpy | 行列演算・スペクトル処理 |
| pandas | データ読み込み・整形（encoding='cp932'） |
| scikit-learn | PLS回帰・GroupKFold・標準化 |
| scipy | Savitzky-Golay フィルタ（savgol_filter） |
| matplotlib | スペクトル・分布の可視化 |
| japanize_matplotlib | matplotlib 日本語フォント対応 |
| lightgbm | 勾配ブースティング（実験用、最終提出には未使用） |

> バージョンは Colab 実行時に `!pip show scikit-learn` 等で確認・記録すること（提出前に補完）

## データファイル

| ファイル | 説明 |
|---------|------|
| `train_near.csv` | 訓練データ（1322件、1559列、cp932） |
| `test_near.csv` | テストデータ（550件、1558列、cp932） |
| `sample_submit_nir.csv` | 提出フォーマットサンプル |

---

# 生成AI利用申告

- **利用ツール**: claude-sonnet-4-6（Anthropic / Claude Code）
- **利用形態**: 有償（Claude Code サブスクリプション）
- **主な利用箇所・用途**:
  - データ探索・前処理設計・モデル選択の実験コード生成
  - CV/LBの乖離診断・失敗要因分析（calibration transfer問題の診断）
  - 文献調査・ドメイン知識（ケモメトリクス）の整理
  - ノートブック（01_EDA〜09_band_cutoff）のコード生成支援

---

# 入賞候補者提出物チェックリスト

## 必須提出物

- [ ] **ソースコード**（学習・前処理部分）
  - `01_EDA.ipynb` / `02_preprocessing.ipynb` / `03_baseline_pls.ipynb`
  - `09_band_cutoff.ipynb` ← **現ベスト LB17.200 の再現**（メイン提出物）
  - ※推論部分（提出CSV生成セル）は除外 or 明示
- [ ] **ソースコード説明書**
  - 前処理部分の説明（SNV、SG微分）
  - 学習部分の説明（PLS、GroupKFold CV）
  - 使用モデルの出所（scikit-learn 標準実装）
- [ ] **実行環境**（本ファイル → Colabバージョン補完後に完成）
- [ ] **データの解釈・工夫点・示唆の説明資料**
  - → `context/handoff_v2.md` の内容をベースに作成
  - calibration transfer問題の診断経緯
  - CV/LBの符号逆転による失敗パターンの実証
  - 樹種非重複制約への対応方針
- [ ] **生成AI利用申告**（上記に記載済み）

## 説明資料に盛り込むべき内容（handoff_v2.md より）

1. **calibration transfer問題の診断**: Train/Test樹種の完全非重複を確認し、種間転移問題と診断した経緯
2. **CV/LBの乖離の構造的説明**: GroupKFold CVがLBより約8〜9低い理由（スギ35%・広葉樹優位）
3. **失敗パターンの実証**: #4で帯域選択のCV最適化がLBを悪化させた事実（過適応の証拠）
4. **SNV+PLS3の選定根拠**: 25通りの前処理×モデル比較、広く粗なモデルが最も転移する理由
5. **文献との接続**: EPO・SNV+d1・sMCがリグノセルロース系NIR研究で繰り返し最善として報告されていること
