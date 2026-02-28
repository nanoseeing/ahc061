- DDP がミスってる
- Log に loss も出したい
- 提出までの 1pass 通す
- bc ができるようにしたい蒸留方法をためしたい
- sps を上げられるか？
- log 出力を増やしたい
- mlflow で結果を見たい
- eval を log に出したい

---

比較レポート正確性の補足 subagent の結果に一部誤りがありました。reinforce には DDP も torch.compile も実装済みです。

reinforce にはなく、exp02 だけにある機能

1. モデルアーキテクチャ機能 exp02 詳細注意機構（Attention） ✓ PlayerSetAttention（プレイヤー間）、MainEnemyCrossAttention（自分 vs 相手）、PixelPlayerSelfAttention（各ピクセルでプレイヤー間） GlobalFiLM ✓ Feature-wise Linear Modulation — global pooling から gamma/beta を生成し、チャンネル毎にアフィン変換（x \* (1+γ) + β） SE Block ✓ Squeeze-and-Excitation — チャンネル重み付きゲート MBConvSE ✓ Mobile Inverted Conv + SE のハイブリッドブロック Depthwise Conv ✓ DWResidualBlock — 軽量な深さ方向畳み込み座標埋め込み ✓ CoordEmbed2d — 盤面上の絶対座標(gx,gy)をチャンネルに追加 PlayerAxisMixResidualBlock ✓ プレイヤー軸方向の混合（identity で初期化）活性化関数 SiLU + LayerNorm reinforce は Tanh 固定モデルの種類 10 種類以上 reinforce は MLP/CNN の 2 種
2. 補助損失（aux loss）の違い exp02 には 2 種類の補助損失があります。reinforce は 1 種類のみ。

補助損失 reinforce exp02 opp_param 損失 MSE（raw のまま） softmax(w[4]) + sigmoid(eps[1]) に通してから MSE — 正規化された空間での MSE opp_move_dist 損失 なし あり — 相手の行動分布（100 マス確率）を KL ダイバージェンスで学習 opp_move_dist の計算（ppo/update.py）：

logp_opp = torch.log_softmax(opp_move_logits, dim=1) loss = -(dist_valid \* logp_opp).sum(dim=1).mean() # KL/cross-entropy 相手の「次の手の分布」を観測から予測させることで、共有表現に相手の意図を埋め込む工夫です。

3. EMA Tracking（過去モデルの保持）

# exp02: 複数の decay 値で並列に EMA モデルを保持

--ema-decays 0.995,0.999 decay=0.999 のモデルは過去 1000 イテレーション程度の平均モデル。評価時・対戦相手として活用でき、方策の安定性と探索の多様性を高めます。reinforce にはない機能です。

4. wandb 完全統合 exp02 は学習ループに直接 wandb.log() が組み込まれています（チェックポイントや動画も wandb に送れる）。reinforce は MLflow 対応のみ。

reinforce だけにある機能（exp02 にない）機能 詳細 VecNormalize 観測値・報酬のランニング正規化（obs_rms, ret_rms） KL 早期停止 target_kl=0.01 等で各エポック後にポリシー変化量を制御豊富なスケジューリング linear, cosine, exp, piecewise, カスタム式 Hydra 設定管理 YAML オーバーライドで実験管理パイプライン BC → PPO → eval の自動化両方に共通する機能機能 DDP（分散学習） torch.compile PPO 基本式（clip surrogate + value clip + entropy） GAE 計算（同一の数値式） action mask 対応 opp_param 補助損失（MSE ベース） C++ BatchEnv 連携 AMP（混合精度）オプション Channels-last memory format

exp002 コマンド vs reinforce exp01.yaml パイプライン 比較

1. ハイパーパラメータの違いパラメータ exp002 reinforce exp01.yaml total timesteps ~256M（5000 更新 × 512env × 100steps） 20M PPO 更新回数 5,000 回 ~391 回 (20M / 51,200) num_envs 512 512 num_steps (t_max) 100 100 learning_rate 3e-4 5e-4 LR warmup なし (0) 1,000,000 steps gamma 1.0 1.0 gae_lambda 0.95 0.99 clip_coef 0.2 0.2 vf_clip_coef 0.2 0.2 ent_coef 0.01 0.01 vf_coef 0.5 1.0 update_epochs 1 1 minibatch_size 512 512 norm_adv ON（バッチ全体） ON（バッチ全体） max_grad_norm - 1.0 aux opp_move 損失 0.05 0（なし） aux opp_param 損失 0.01 0（なし）
2. モデルの違い項目 exp002 (dwres_v1, hidden=128, blocks=16) reinforce (TeacherP0V1BoardAgent) Stem Conv1×1 → GroupNorm → SiLU Conv → ResBlock 前処理本体ブロック 16 × DWResidualBlock 4 × ResBlock 畳み込み種類 Depthwise-Separable（DW 3×3 + PW 1×1） 標準 Conv3×3 活性化関数 SiLU tanh 隠れ層チャンネル 128 64 Value head GAP（シンプル） GAP → Linear[128] → Linear[64] → Linear[1] モデル規模 大（128ch × 16 ブロック、DW-Conv で軽量化） 小（64ch × 4 ブロック）正規化 GroupNorm GroupNorm
3. 特徴量の違い項目 exp002 (research_v3, 67ch) reinforce (teacher_p0_v1_88ch, 88ch) チャンネル数 67 88 プレイヤー表現 全 8 プレイヤーを対称的に並列表現 自分(11ch) + 敵 7 人(11ch×7)、スコア降順 m/u エンコード one-hot（m: 7ch, u: 5ch） スカラー正規化（m_base, u_base）相手パラメータ なし ベイズ推定値(wb,wc,wd,eps)を直接含む距離情報 dist_owner, dist_comp, dist_center 自分・敵それぞれの dist_from_piece, dist_to_reach 次ターン予測 next_prob（各プレイヤー） next_prob（自分+各敵）粒子フィルタ なし（特徴量に組み込まない） オンライン Bayes で計算、特徴量に直接埋め込む重要な設計思想の違い:

research_v3: 全プレイヤー平等・対称な表現。opponent modeling は aux 損失で間接的に学習 teacher_p0_v1_88ch: 自己中心的（ego-centric）表現。粒子フィルタによる推定値を特徴量として直接入力 4. PPO 学習の工夫の違い工夫 exp002 reinforce Auxiliary 損失 opp_move_dist (KL, 0.05) + opp_param (MSE, 0.01) なし相手モデリング aux 損失でネットワークに学習させる 粒子フィルタ（Bayes）で計算、特徴量に入力 LR warmup なし 最初の 1M ステップで線形ウォームアップ norm_adv ON（バッチ全体） ON（バッチ全体） AMP なし（コマンド未指定） なし (ppo_amp: false) torch.compile なし あり (ppo_compile: true) 分散学習 なし auto 検出 (ppo_distributed: auto) 評価 なし（訓練のみ） 2M ステップごと・最終評価あり BC フェーズ なし skip_bc: true（スキップ中） 5. その他の違い項目 exp002 reinforce 粒子フィルタ なし ppo_pf_enabled: true（全環境でリアルタイム計算） MLflow 追跡 なし 対応あり学習規模 12.8 倍多いタイムステップ 小規模（プロトタイプ設定） gae_lambda 0.95 (短期有利) 0.99 (長期有利) vf_coef 0.5 (standard) 1.0 (強めの VF 学習) まとめ：最も大きな違い学習量: exp002 は 256M steps vs reinforce 20M steps（12.8 倍差）モデル規模: exp002 は 128ch×16 ブロック vs reinforce 64ch×4 ブロック（大幅に大きい）相手モデリング手法: exp002 は aux 損失で内部化、reinforce は粒子フィルタで外部計算して特徴量入力 gae_lambda: 0.95 vs 0.99（reinforce の方が長期リターンを重視） vf_coef: 0.5 vs 1.0（reinforce の方が価値関数の学習を強調）
