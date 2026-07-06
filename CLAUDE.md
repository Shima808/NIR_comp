# NIR木材含水率予測コンペ（Signate）

## 🏁 コンペ終了（2026-07-06 結果確定）

- **最終順位: トップ20圏外**。ソースコード提出はトップ10のみ義務 → **提出義務なし（確認済み）**
- 最終選択はPublic最良の2つ（Private 13.79/13.80）。**選ばなかった素のSNV+PLS3がPrivate 9.780で13〜15位相当**だった
- 敗因: Public LBへの過適応（5500カットオフはPublicだけに効き、Privateで悪化）。CVは正解を指していた（詳細は下記・振り返りは [context/retrospective.md](context/retrospective.md)）
- **残タスク: なし**（提出不要。振り返りメモのみ保存済み）

## コンペの核心

- **タスク**: NIRスペクトル(1555波長) → 含水率(%) の回帰、評価RMSE
- **締切**: 2026-06-30
- **最重要制約**: Train(13樹種) と Test(6樹種) が**完全に別種** → calibration transfer問題
- `species number`・樹種列は**絶対に特徴量に使わない**

## 現在の最良提出

- **LB: 17.200** — SNV(全波長) → 5500 cm⁻¹以上を選択 → PLS(n=3) → clip[0.84, 298.58]
- LOSO mean: 22.45 / median: 13.78
- 物理的根拠: O-7論文「5200 cm⁻¹付近の飽和帯域除去」

## CV/LBの法則（実証済み）

- GroupKFold CVはLBより約−8〜9程度低め（スギ35%=広葉樹優位のため）
- **CVを下げてもLBが下がるとは限らない**（#4でCV改善→LB悪化を確認）
- チューニングはGroupKFold/LOSOで行い、LBは1回だけ確認に使う

## 禁止事項（再現禁止の失敗パターン）

1. 樹種 or species numberを特徴量化
2. random KFold（同一樹種がtrain/valに混在してリーク）
3. GroupKFold/LOSOでの帯域・ハイパーパラメータのCV最適化探索

## 試行済み・確定失敗（再試不要）

- **EPO**（種平均/乾燥曲線、全波長/5500以上）→ 全バリアント悪化
- **EMSC/MSC** → SNVより悪化
- **SVR/LightGBM等の非線形モデル** → 訓練樹種に過剰適合、LB悪化
- **帯域アンサンブル**（5400+5500）→ LOSO改善もLB悪化
- **物理特徴量**（ratio_1900/1450）→ 誤差範囲内

## 残りアクション → 全て終了（2026-07-06）

- ~~賞与資料~~ → **不要**（トップ10圏外で提出義務なし）。代わりに軽い振り返りを [context/retrospective.md](context/retrospective.md) に保存
- ~~コード3分割整備~~ → **完了**: `src/preprocessing.py` / `learning.py` / `predicting.py`（自分の資産として保持。`python learning.py`→`models/pls3_5500up.joblib`→`python predicting.py`）
- モデル改善・追加提出 → **終了**（コンペ終了）

## ⚠️ ルール上禁止と確認された手法（2026-06-22）

- **PDS/SST/NS-PFCE** — testスペクトル複数をまとめて解析する転移補正 → **ルール違反・実施不可**
  （「評価用データ同士の関係を利用した補正」に該当）

## フォルダ構成

```
NIR.comp/
├── CLAUDE.md              ← このファイル
├── data/                  ← データファイル（cp932エンコード）
│   ├── train_near.csv
│   ├── test_near.csv
│   └── sample_submit_nir.csv
├── context/               ← 引き継ぎ・設計ドキュメント
│   └── handoff_v2.md
├── literature/            ← 論文PDF・.md化した文献知識
├── notebooks/             ← Colab用ipynb
│   ├── 01_EDA.ipynb
│   └── 02_preprocessing.ipynb
├── src/                   ← 提出用3分割モジュール（ベスト再現）
│   ├── preprocessing.py   ← SNV＋5500cm⁻¹以上カットオフ
│   ├── learning.py        ← PLS3学習＋CV/LOSO診断＋モデル保存
│   └── predicting.py      ← モデル読込＋予測＋提出CSV出力
├── models/                ← 学習済みモデル（pls3_5500up.joblib）
└── submissions/           ← 提出CSVの履歴
```

## 提出コード実行手順（src/）

```bash
cd src
python learning.py     # 診断表示＋ models/pls3_5500up.joblib を保存
python predicting.py   # モデル読込→ submissions/submission_pls3_5500up.csv 生成
```

## データパス（Colab実行時）

```python
READ_PATH = '/content/train (2).csv'   # encoding='cp932'
SAVE_PATH = '/content/submission.csv'  # Driveではなく/content/に保存
DATA_DIR  = 'C:/Users/enzos/Downloads/NIR.comp/data/'  # ローカル参照用
```

## 詳細な実験履歴・LOSOスコア・文献情報

→ [context/handoff_v2.md](context/handoff_v2.md) を参照
