# EEG Classification with Stacked Ensemble

## 概要

本リポジトリは、以下の最終課題への提出用スクリプトです。
- 最終課題: 被験者が画像を見ているときの脳波 (EEG) から，その画像がどのクラスに属するかを分類するタスク。

主に以下2つのモデルを学習・アンサンブルし、さらに LightGBM を用いたメタモデル（スタッキング）で最終予測を生成します。
- **TransformerClassifier**  
  1D-CNN (EEGNetBlock) + TransformerEncoder の組み合わせ  
- **CNN_LSTM_Classifier**  
  EEGNetBlock＋AdaptiveAvgPool1d + 双方向LSTM の組み合わせ  

最終的に、各モデルの softmax 出力を特徴量として LightGBM に学習させ、メタ予測器として活用します。

## 実装の流れ

1. ### データセットへの前処理  
   - **RawEEGDataset**：生波形を直接返すクラス。  
   - 訓練時のデータ拡張 (MixUp、可変ノイズ、ゲインスケーリング、チャネルフリップなど) を実装。

2. ### モデル定義  
   - **EEGNetBlock**：Temporal → Depthwise → Pointwise の 1D Conv ブロック。  
   - **TransformerClassifier**：EEGNetBlock → AdaptiveAvgPool1d → Positional Encoding → TransformerEncoder → 平均プーリング → 全結合  
   - **CNN_LSTM_Classifier**：EEGNetBlock → AdaptiveAvgPool1d → 双方向 LSTM → 平均プーリング → 全結合  

3. ### 学習ループ (`train_model` 関数)  
   - **One-Cycle LR** (`torch.optim.lr_scheduler.OneCycleLR`) で学習率をバッチ単位で変動制御。  
   - **MixUp** を導入し、バッチ内サンプル同士を線形結合してラベルも混合。  
   - TensorBoard ログ出力に対応。  
   - 各エポックで検証精度が更新された場合に `*_best.pt` としてモデル重みを保存。

4. ### スタッキング (メタモデル)  
   - 検証セット上で両モデルの softmax 確率を取得し、特徴行列 `X_meta` を構築。  
   - **LightGBM** をメタモデルとして訓練し、検証セットでの Top-1 Accuracy を大幅改善。  

5. ### 推論 & 提出ファイル作成  
   - テストセットで同様にメタ特徴量を作成し、LightGBM で予測確率を出力。  
   - 予測確率は、`submission.npy` に保存し、最良重みの `transformer_best.pt`／`cnnlstm_best.pt` 及び Notebook と共に `submission.zip` を作成。

## 実装での工夫点

- **データ拡張 (Augmentation)**  
  - **MixUp**: バッチ内の二つのサンプルを線形結合し、ラベルも混合して訓練データを拡張。  
  - **可変ガウシアンノイズ**: 訓練時にσ ∈ [0.005, 0.02]からノイズ強度をランダムサンプリングして付加。  
  - **ゲインスケーリング**: 信号振幅をランダムに0.9〜1.1倍して、振幅変化に頑健に学習。  
  - **チャネルフリップ**: 50%の確率でチャネル順序を反転し、左右脳波のシンメトリに対応。  
  - **チャネルドロップ**: 各チャネルを10%の確率でゼロマスクし、特定チャネルへの過度な依存を抑制。  
  - **時間マスク (SpecAugment風)**: 時間軸上のランダム区間（全長の20%以内）をゼロクリップし、時間的頑健性を向上。  
- **One-Cycle Learning Rate**  
  学習率を三角状に変動させることで、早期に大きな探索を行いつつ、最後に収束を促進させる。
- **MixUp Augmentation**  
  バッチサイズを有効活用し、過学習を抑制しながらモデルの汎化性を向上。
- **Stacking with LightGBM**  
  単純アンサンブルより複雑な相互作用を学習できるメタモデルで、検証精度(val acc)を約0.52まで引き上げた。
- **学習スキップ機能**  
  既存の重みファイルが存在する場合、自動的に学習フェーズをスキップし、推論フェーズに直接移行するフラグ `SKIP_TRAIN` を実装。開発・デバッグが効率化した。
- **グリッドサーチによるアンサンブル重み最適化 (最終コードではコメントアウト)**  
  検証セット上で粗いグリッドサーチと微細グリッドサーチを2段階で実施し、最適なモデル重み平均係数を自動探索。
- **モジュール化された学習・評価パイプライン**  
  `train_model` 関数に学習・検証ロジックを集約し、異なるモデルや拡張手法の組み替えが容易にした。

---

## 提出物

- テストデータに対する予測確率：submission.npy
- 本 Notebook（`DLBasics2025_competition_stacking.ipynb`）  
- 重みファイル：Transformer と CNN-LSTM の最良重み (`transformer_ft_best.py`, `ccnlstm_best.pt`)  
- 以上を `submission.zip` にまとめて提出。  
>>>>>>> 93ac954 (Add EEG classification code and README)
