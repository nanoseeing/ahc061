# AHC061 PPO モデル設計案

最終更新: 2026-02-25

この文書は、これまでの議論を「実装順序が決められる粒度」に整理したモデル設計メモです。  
対象は `reinforce/ppo_discrete` の AHC061 学習系です。

## 0. 記号と優先度

### 記号

- 盤面セル: `c=(x,y)`, `x,y in {0..9}`
- プレイヤー: 自分 `p=0`, 敵 `p in {1..M-1}`, `M<=8`
- 敵スロット: `e in {0..6}`（`player_id=e+1`。存在しないスロットは 0 埋め）
- 領土所有者: `O(c) in {-1, 0, ..., M-1}`
- レベル: `L(c) in {0,1,...,U}`
- 価値: `V(c)`
- 自分/敵駒位置: `pos_0`, `pos_e`
- 敵次手分布: `A_e(c)=P(d_e=c | history)`（Bayes推定に基づく）
- スコア: `S_p = sum_c V(c) * L(c) * 1[O(c)=p]`
- 最大敵: `e* = argmax_{p>=1} S_p`
- 目的値: `J = log2(S_0 / max_{p>=1} S_p)`

### 優先度

- `P0`: 1回目で必須。費用対効果が高い。
- `P1`: 2回目で導入。改善余地が高い。
- `P2`: 効果はあるが重くなりやすい。検証優先。
- `P3`: 実験枠。最後に回す。

## 0.5 優先度0 教師モデル案（Global FC排除・非集約）

この節は、次に試す「優先度0の教師モデル」を固定仕様として定義する。  
方針は以下。

- `global` ベクトルは使わず、使う情報はすべて `10x10` チャネルへ落とし込む（broadcast含む）。
- 敵の情報は集約せず、敵スロットごとのチャネルをそのまま与える。
- 敵スロットは「敵領土合計（降順）」で並べる（同点時は `player_id` 昇順で安定化）。

### 0.5.1 入力テンソル仕様

- 形状: `B x C x 10 x 10`
- 総チャネル数: `C=88`
  - global相当 (broadcast/盤面系): `5ch`
  - 自分: `6ch`
  - 敵: `7 slot x 11ch = 77ch`

補助記号:

- `pi(e)`: 敵スロット `e in {0..6}` に対応する実敵 `player_id`
- `TerrScore_p = sum_c V(c)*L(c)*1[O(c)=p]`
- `pi(e)` は `p in {1..M-1}` を `TerrScore_p` 降順・`player_id` 昇順で並べた `e` 番目
- `e >= M-1` の空きスロットは全チャネルを0埋め

### 0.5.2 チャネル定義

#### A. Global相当（盤面化）

| 変数名 | ch数 | 意味 | 数式/定義 |
|---|---:|---|---|
| `g_turn_norm` | 1 | ターン進行度 | `t/T` を全セルへ broadcast |
| `g_m_norm` | 1 | プレイヤー数正規化 | `(M-2)/6` を全セルへ broadcast |
| `g_u_norm` | 1 | レベル上限正規化 | `(U-1)/4` を全セルへ broadcast |
| `value_norm` | 1 | セル価値正規化 | `V(c)/1000` |
| `owner_neutral` | 1 | 未所有セル one-hot | `1[O(c)=-1]` |

#### B. 自分

| 変数名 | ch数 | 意味 | 数式/定義 |
|---|---:|---|---|
| `self_level_norm` | 1 | 自領土のlv正規化 | `(L(c)/U) * 1[O(c)=0]` |
| `self_pos` | 1 | 自駒座標 | `1[c=pos_0]` |
| `self_reachable` | 1 | 自駒を含む自領土連結の到達可能 | `1[c in Reach_0]` |
| `self_frontier` | 1 | 自連結領域の外周 | `1[c notin Reach_0 and N4(c) intersects Reach_0]` |
| `self_dist_from_piece` | 1 | 自駒からの距離 | `clip(L1(c,pos_0),0,18)/18` |
| `self_dist_to_reachable` | 1 | 自連結領域までの距離 | `clip(min_{r in Reach_0} L1(c,r),0,10)/10` |

#### C. 敵（スロットごと、`e=0..6`）

| 変数名 | ch数 | 意味 | 数式/定義 |
|---|---:|---|---|
| `enemy_level_norm_e` | 7 | 敵領土lv正規化 | `(L(c)/U) * 1[O(c)=pi(e)]` |
| `enemy_pos_e` | 7 | 敵駒座標 | `1[c=pos_{pi(e)}]` |
| `enemy_reachable_e` | 7 | 敵駒を含む敵領土連結の到達可能 | `1[c in Reach_{pi(e)}]` |
| `enemy_frontier_e` | 7 | 敵連結領域の外周 | `1[c notin Reach_{pi(e)} and N4(c) intersects Reach_{pi(e)}]` |
| `enemy_dist_from_piece_e` | 7 | 敵駒からの距離 | `clip(L1(c,pos_{pi(e)}),0,18)/18` |
| `enemy_dist_to_reachable_e` | 7 | 敵連結領域までの距離 | `clip(min_{r in Reach_{pi(e)}} L1(c,r),0,10)/10` |
| `enemy_next_prob_e` | 7 | 敵次手確率分布 | `A_{pi(e)}(c)` |
| `bayes_wb_e` | 7 | Bayes推定 `wb/wa`（正規化） | `((clip(hat(wb/wa),R_LO,R_HI)-R_LO)/(R_HI-R_LO))` を全セルへ broadcast |
| `bayes_wc_e` | 7 | Bayes推定 `wc/wa`（正規化） | `((clip(hat(wc/wa),R_LO,R_HI)-R_LO)/(R_HI-R_LO))` を全セルへ broadcast |
| `bayes_wd_e` | 7 | Bayes推定 `wd/wa`（正規化） | `((clip(hat(wd/wa),R_LO,R_HI)-R_LO)/(R_HI-R_LO))` を全セルへ broadcast |
| `bayes_eps_e` | 7 | Bayes推定 `eps`（正規化） | `((clip(hat(eps),E_LO,E_HI)-E_LO)/(E_HI-E_LO))` を全セルへ broadcast |

### 0.5.3 情報網羅性チェック

この構成は「盤面・敵の状態情報」をほぼ網羅できるが、以下を明示しておく。

| 観点 | 判定 | 補足 |
|---|---|---|
| 盤面価値/未所有/領土lv/駒位置 | ほぼ網羅 | 入っている |
| 敵ごとの局所構造（連結・外周・距離） | 網羅 | 7スロットで個別入力 |
| 敵の次手分布とBayes推定 | 網羅 | `A_e` + `wb/wc/wd/eps` を個別保持 |
| 敵IDの時間一貫性 | 注意 | 「領土合計ソート」はターンごとにスロット入替が起こる |
| 合法手情報 | 別経路で必要 | `use_action_mask=true` を前提にするのが安全 |
| スコア文脈（`S0`, `SA`, top敵） | 暗黙的には推定可能 | 明示入力ではないため学習難度は上がる |

推奨補足（P0+α）:

- `use_action_mask=true` を必須運用にする（合法手をlogits側で厳密制約）。
- 敵スロット並びは同点タイブレークを固定し、チャネル間対応の揺れを抑える。
- もし学習が不安定なら、次に `top_enemy_owner` と `A_top` を追加して比較する。

### 0.5.4 CNN構造定義（優先度0教師）

実装は既存の `StudentMBoardAgent` を再利用し、`global_dim=0` で「盤面のみCNN」にする。

推奨の初期教師構成（Teacher-P0-v1）:

| 項目 | 設定値 |
|---|---|
| `type` | `StudentMBoardAgent` |
| `board_channels` | `88` |
| `board_size` | `10` |
| `global_dim` | `0` |
| `width` | `64` |
| `num_blocks` | `4` |
| `value_hidden_dims` | `[128, 64]` |
| `activation` | `tanh` |
| `use_global_film` | `false` |
| `use_global_policy_bias` | `false` |

ネットワーク構造:

1. Input: `B x 88 x 10 x 10`
2. Stem: `Conv3x3(88->64)` + `Tanh`
3. Residual Block x4:
   - `Conv3x3(64->64)` + `Tanh`
   - `Conv3x3(64->64)`
   - Skip add + `Tanh`
4. Policy Head: `Conv1x1(64->1)` -> flatten -> `100 logits`
5. Value Head: `GAP(10x10)` -> `Linear(64->128)` -> `Tanh` -> `Linear(128->64)` -> `Tanh` -> `Linear(64->1)`

### 0.5.5 パラメータ数（88ch入力時）

以下は `StudentMBoardAgent(global_dim=0, board_channels=88, board_size=10)` の実測パラメータ数。

| 用途 | `width` | `num_blocks` | `value_hidden_dims` | パラメータ数 |
|---|---:|---:|---|---:|
| 軽量教師 | 48 | 3 | `[64]` | 166,018 |
| 標準教師 | 64 | 4 | `[64]` | 350,466 |
| 推奨教師（v1） | 64 | 4 | `[128, 64]` | 362,882 |
| 大型教師 | 80 | 4 | `[64]` | 530,210 |
| 超大型教師 | 96 | 4 | `[64]` | 746,818 |

備考:

- `value_hidden_dims=[64]` のとき、概算式は  
  `P = 18*B*W^2 + (858 + 2*B)*W + 130`（`W=width`, `B=num_blocks`）。
- まず `Teacher-P0-v1` で学習を開始し、計算資源に余裕があれば `width=80` を比較する。

実装対応（2026-02-25）:

- 環境特徴量: `AHC061LocalEnv(obs_mode="teacher_p0_v1_88ch")`
- モデル設定: `reinforce/configs/ppo_discrete/model/model.ahc061_teacher_p0_v1.toml`
- パイプライン設定:
  - `reinforce/configs/ppo_discrete/pipeline/run_pipeline.ahc061_teacher_p0_v1_light.toml`
  - `reinforce/configs/ppo_discrete/pipeline/run_pipeline.ahc061_teacher_p0_v1_full.toml`

## 1. 盤面チャネル総覧

注: `*_e` は敵スロットごと（最大7枚）を意味します。

| 変数名 | ch数 | 意味 | 数式/定義 | 優先度 |
|---|---:|---|---|---|
| `value_norm` | 1 | セル価値正規化（盤面間で固定） | `V(c)/1000` | P0 |
| `owner_self` | 1 | 自領土 | `1[O(c)=0]` | P0 |
| `owner_enemy_any` | 1 | 敵領土（敵ID無視） | `1[O(c)>=1]` | P0 |
| `owner_neutral` | 1 | 未所有 | `1[O(c)=-1]` | P0 |
| `level_norm` | 1 | レベル正規化 | `L(c)/U` | P0 |
| `piece_self` | 1 | 自駒位置 | `1[c=pos_0]` | P0 |
| `piece_enemy_any` | 1 | 敵駒位置（敵ID無視） | `(sum_e 1[c=pos_e]) / max(1, M-1)` | P0 |
| `legal_mask_self` | 1 | 自分の合法手（`use_action_mask` 未使用時のみ） | `1[c in Cand_0]` | P2 |
| `reachable_self` | 1 | 自領土連結で到達可能 | `1[c in Reach_0]` | P0 |
| `frontier_self` | 1 | 自領土外周 | `1[c notin Reach_0 and N4(c) intersects Reach_0]` | P0 |
| `dist_from_self` | 1 | 自駒からの距離 | `clip(L1(c,pos_0),0,18)/18` | P1 |
| `dist_to_reachable_self` | 1 | 自連結領域までの距離 | `clip(min_{r in Reach_0} L1(c,r),0,10)/10` | P1 |
| `x_coord` | 1 | x座標埋め込み | `x/9` | P2 |
| `y_coord` | 1 | y座標埋め込み | `y/9` | P2 |
| `self_prev_pos` | 1 | 前ターン自駒位置 | `1[c=pos_0(t-1)]` | P2 |
| `enemy_pos_e` | 7 | 敵eの駒位置 | `1[c=pos_e]` | P1 |
| `enemy_territory_e` | 7 | 敵eの領土 | `1[O(c)=e+1]` | P1 |
| `enemy_level_e` | 7 | 敵e領土レベル | `(L(c)/U)*1[O(c)=e+1]` | P2 |
| `enemy_piece_territory_e` | 7 | 敵e駒を含む連結領域 | `1[c in CC_e(pos_e)]` | P2 |
| `enemy_piece_frontier_e` | 7 | 上記領域の外周 | `1[c notin CC_e and N4(c) intersects CC_e]` | P2 |
| `A_enemy_e` | 7 | 敵eの次手確率分布 | `A_e(c)` | P1 |
| `enemy_rb_on_territory_e` | 7 | `rb` を敵領土に塗布 | `r^b_e * 1[O(c)=e+1]` | P3 |
| `enemy_rc_on_territory_e` | 7 | `rc` を敵領土に塗布 | `r^c_e * 1[O(c)=e+1]` | P3 |
| `enemy_rd_on_territory_e` | 7 | `rd` を敵領土に塗布 | `r^d_e * 1[O(c)=e+1]` | P3 |
| `enemy_rb_on_piece_territory_e` | 7 | `rb` を駒連結領域に塗布 | `r^b_e * 1[c in CC_e(pos_e)]` | P3 |
| `enemy_rc_on_piece_territory_e` | 7 | `rc` を駒連結領域に塗布 | `r^c_e * 1[c in CC_e(pos_e)]` | P3 |
| `enemy_rd_on_piece_territory_e` | 7 | `rd` を駒連結領域に塗布 | `r^d_e * 1[c in CC_e(pos_e)]` | P3 |
| `E_in` | 1 | 期待侵入人数 | `sum_e A_e(c)` | P0 |
| `P_any` | 1 | 誰かが来る確率 | `1 - prod_e (1 - A_e(c))` | P0 |
| `A_max` | 1 | 最大敵到達確率 | `max_e A_e(c)` | P1 |
| `A_entropy` | 1 | 到達分布の敵方向エントロピー | `-sum_e p_e(c)log(p_e(c)+eps)` (`p_e=A_e/sum_j A_j`) | P2 |
| `A_top` | 1 | 現在トップ敵の到達分布 | `A_{e*}(c)` | P0 |
| `top_enemy_owner` | 1 | トップ敵領土 | `1[O(c)=e*]` | P0 |
| `top_enemy_level` | 1 | トップ敵領土レベル | `(L(c)/U)*1[O(c)=e*]` | P1 |
| `A_top2` | 1 | 2位敵の到達分布 | `A_{e2}(c)` | P2 |
| `A_other` | 1 | 上位以外の到達分布合計 | `sum_{e notin {e*,e2}} A_e(c)` | P2 |
| `enemy_atk_territory` | 1 | 攻撃性重み付き敵領土 | `sum_e atk_e * 1[O(c)=e+1]` | P0 |
| `danger_arrival` | 1 | 攻撃性重み付き到達危険度 | `sum_e atk_e * A_e(c)` | P1 |
| `enemy_grow_territory` | 1 | 成長性重み付き敵領土 | `sum_e grow_e * 1[O(c)=e+1]` | P1 |
| `gain_if_capture` | 1 | 奪取時の目的値改善量 | `clip((J_cap(c)-J_now)/s_gain, -1, 1)` | P0 |
| `loss_if_lost` | 1 | 自領土喪失時の目的値悪化量（期待値） | `clip((J_now-J_lose_exp(c))/s_loss, 0, 1)` | P0 |
| `contest_risk` | 1 | 競合危険度 | `frontier_self(c) * P_any(c)` | P0 |
| `self_territory_value` | 1 | 自領土価値マスク | `(V(c)/1000)*1[O(c)=0]` | P1 |

補助定義（推奨）:

- `atk_e = clip(sigmoid(a_b*r^b_e + a_c*r^c_e + a_d*r^d_e - a_eps*eps_e), 0, 1)`
- `grow_e = clip(b_s*score_ratio_e + b_f*frontier_ratio_e + b_a*atk_e, 0, 1)`
- `J_now = log2(S_0 / S_A)`, `S_A = max_{p>=1} S_p`
- `J_cap(c)`: 「セル `c` を今奪取/強化できた」と仮定した 1 手後 `log2(S'_0 / S'_A)`
  - 例（敵 `p` 領土を奪取）: `S'_0=S_0+v`, `S'_p=S_p-v`, `S'_A=max(S'_p, S_second)` if `p=top` else `S_A`
- `J_lose_exp(c)`: 「自領土 `c` を敵に取られる」場合の期待 `log2(S'_0 / S'_A)`
  - `J_lose_exp(c)=sum_e w_e(c) * J_lose(c,e)`（`w_e` は `A_e(c)` か正規化版）
- `s_gain`, `s_loss`: スケール定数（例: `0.25`）

P1の初手実装は次を推奨（最小セット）:

- `enemy_pos_e`（7ch）
- `enemy_territory_e`（7ch）
- `A_enemy_e`（7ch）
- `A_max`（1ch）
- `top_enemy_level`（1ch）
- `self_territory_value`（1ch）
- `danger_arrival`（1ch）

## 2. Global パラメータ総覧

`global` は「空間に落としにくい情報」だけを残すのが原則です。  
空間に落とせるもの（危険マップ/領土マップ等）は盤面チャネル化を優先します。

| 変数名 | 次元 | 意味 | 数式/定義 | 優先度 |
|---|---:|---|---|---|
| `turn_norm` | 1 | ターン進行度 | `t/T` | P0 |
| `m_norm` | 1 | プレイヤー数正規化 | `(M-2)/6` | P0 |
| `u_norm` | 1 | レベル上限正規化 | `(U-1)/4` | P0 |
| `self_x_norm`, `self_y_norm` | 2 | 自駒座標 | `x_0/9, y_0/9` | P0 |
| `enemy_x_norm_e`, `enemy_y_norm_e` | 14 | 敵座標（固定7スロット。`enemy_pos_e` を使うなら削除候補） | `x_e/9, y_e/9` | P2 |
| `self_score_norm` | 1 | 自スコア正規化 | `S_0/S_cap` | P0 |
| `max_enemy_score_norm` | 1 | 最大敵スコア正規化 | `max_{p>=1} S_p / S_cap` | P0 |
| `enemy_alive_e` | 7 | 敵スロット有効フラグ | `1[e+1 < M]` | P1 |
| `enemy_score_norm_e` | 7 | 敵ごとのスコア | `S_{e+1}/S_cap` | P1 |
| `enemy_score_gap_e` | 7 | 自分との差 | `(S_0 - S_{e+1})/S_cap` | P1 |
| `top_enemy_onehot` | 7 | 現トップ敵ID | `one_hot(e*)` | P1 |
| `top_enemy_score_norm` | 1 | トップ敵スコア | `S_{e*}/S_cap` | P1 |
| `second_enemy_score_norm` | 1 | 2位敵スコア | `S_{e2}/S_cap` | P2 |
| `score_ratio_s0_sa` | 1 | 比率目的の現在値 | `S_0/max_{p>=1}S_p` | P1 |
| `bayes_rb_e` | 7 | Bayes推定 `wb/wa` | `r^b_e` | P0 |
| `bayes_rc_e` | 7 | Bayes推定 `wc/wa` | `r^c_e` | P0 |
| `bayes_rd_e` | 7 | Bayes推定 `wd/wa` | `r^d_e` | P0 |
| `bayes_eps_e` | 7 | Bayes推定 `eps` | `eps_e` | P0 |
| `bayes_std_rb_e` | 7 | `rb` 不確実性 | `std(rb_e)` (粒子分散) | P2 |
| `bayes_std_rc_e` | 7 | `rc` 不確実性 | `std(rc_e)` | P2 |
| `bayes_std_rd_e` | 7 | `rd` 不確実性 | `std(rd_e)` | P2 |
| `bayes_std_eps_e` | 7 | `eps` 不確実性 | `std(eps_e)` | P2 |
| `legal_count_norm` | 1 | 自合法手数 | `|Cand_0|/100` | P1 |
| `objective_prev` | 1 | 1ターン前目的値 | `J_{t-1}` | P2 |
| `objective_delta_prev` | 1 | 1ターン差分 | `J_{t-1}-J_{t-2}` | P2 |

備考:

- `use_action_mask=true` を使う構成では、`legal_mask_self` の入力は基本不要。
- `enemy_pos_e` を盤面入力する場合、`enemy_x/y_norm_e` は冗長になりやすい（Phase0で削除アブレーション推奨）。

## 3. モデル構造の工夫（網羅）

| 手法 | 目的 | 実装案 | コスト | 優先度 |
|---|---|---|---|---|
| `CNN + FC`（現行系） | 安定な基準線 | 盤面CNN + global連結 + actor/critic head | 低 | P0 |
| `Action Mask` | 非合法手排除 | logits の invalid を `-inf` | 低 | P0 |
| `Observation Normalization` | スケール安定化 | channel/global ごとに [0,1] 正規化 | 低 | P0 |
| `Residual Conv Block` | 表現力向上 | Conv-Tanh を ResNet化 | 中 | P1 |
| `SE / Channel Gating` | 有効チャネル強調 | グローバルプーリングで channel reweight | 低 | P1 |
| `FiLM (enemy-conditioned)` | 敵パラメータと盤面の対応付け | `A'_e = gamma(w_e)*A_e + beta(w_e)` | 低 | P1 |
| `敵方向 Attention` | 敵集合の順序不変集約 | query=盤面要約, key/value=敵特徴 | 中 | P2 |
| `Top-K Slot + Others` | トップ敵/次点敵の明示 | `A_top`, `A_top2`, `A_other` を固定化 | 低 | P1 |
| `DeepSets Encoder` | 敵数可変への頑健化 | `phi(enemy_e)` を `sum/max` 集約 | 低 | P2 |
| `Auxiliary Heads` | 学習信号追加 | top敵予測/危険マップ再構成の補助loss | 中 | P2 |
| `Dueling Critic / Distributional Value` | 価値推定改善 | Value head を分解または分布化 | 中 | P3 |

### 3.1 具体案: 「近づいてはいけない領土位置」を作る

1. 敵ごとに攻撃性を算出:
   - `atk_e = f(r^b_e, r^c_e, r^d_e, eps_e)`
2. 領土と混合:
   - `danger_territory(c) = sum_e atk_e * 1[O(c)=e+1]`
3. 敵次手分布を混合:
   - `danger_arrival(c) = sum_e atk_e * A_e(c)`
4. 最終危険マップ:
   - `danger(c) = lambda1*danger_territory(c) + lambda2*danger_arrival(c) + lambda3*P_any(c)`
5. 利用方法:
   - 入力チャネルに追加（推奨）
   - または policy trunk 内で FiLM / attention の重み付けに利用

## 4. 提出制約（512KiB / 2sec）を踏まえた上限設計

### 4.1 モデル容量の実用上限

提出制約: ソースサイズ `<= 512 KiB`。  
モデルを埋め込む場合は、以下で上限を見積もる。

- `S_limit = 524,288` bytes
- `S_code`: 推論コード + 既存ロジック（実装次第だが 150-200KiB を想定）
- `S_margin`: 予備（20KiB 推奨）
- `S_blob_text = S_limit - S_code - S_margin`
- `N_max ~= S_blob_text / (enc_overhead * bytes_per_param * comp_ratio)`

ここで:

- `enc_overhead`: テキスト化のオーバーヘッド（base64 は `4/3`）
- `comp_ratio`: 圧縮後サイズ / 生データサイズ（無圧縮なら `1.0`、圧縮で小さくなるほど有利）

`S_code=170KiB`, `S_margin=20KiB` を仮定した目安:

| 保存形式 | 前提 | 実用 `N_max` 目安 | コメント |
|---|---|---:|---|
| `fp32 + base64` | `4 byte/param`, `comp_ratio=1.0` | `~55k` | 容量的に厳しい |
| `fp16 + base64` | `2 byte/param`, `comp_ratio=1.0` | `~110k` | 容量は通るが CPU 速度メリットは小さい |
| `int8 + base64` | `1 byte/param`, `comp_ratio=1.0` | `~220k` | 提出の主力候補 |
| `int8 + 圧縮 + base64` | `1 byte/param`, `comp_ratio=0.7` | `~315k` | 圧縮器/復号器のコード量に注意 |
| `int4 pack + base64` | `0.5 byte/param`, `comp_ratio=1.0` | `~440k` | 実装難度が上がる |

実運用での推奨:

- 安全側: `int8` で `120k-180k params` に抑える。
- 攻める場合: `int8` で `200k-250k params`（圧縮・コードサイズ管理が必須）。

### 4.2 圧縮方式の優先順位

| 手法 | 容量効果 | 速度効果 | 実装難度 | 推奨度 |
|---|---|---|---|---|
| `int8量子化 (per-channel)` | 大 | 大 | 中 | 最優先 |
| `蒸留 (teacher->student)` | 中-大 | 中-大 | 中 | 最優先 |
| `構造削減 (チャネル/層削減)` | 大 | 大 | 低 | 最優先 |
| `fp16化` | 中 | 小-中 | 低 | 補助 |
| `int4量子化` | 大 | 中 | 高 | 余力があれば |
| `Huffman/LZ系圧縮` | 中 | なし | 中-高 | サイズ不足時のみ |
| `16進数文字列埋め込み` | 悪い | なし | 低 | 非推奨 |
| `10進数配列埋め込み` | 非常に悪い | なし | 低 | 非推奨 |

注意:

- 512KiB 制約は「ソース文字列サイズ」なので、重みのテキスト表現効率が最重要。
- `base64` は実装が簡単で効率も良い。`hex/decimal` は非推奨。

### 4.3 推論時間目標（2sec / 100turn）

制約から 1 ターンあたり全処理予算は `20ms`。  
NN 推論以外（状態更新・特徴量生成・Bayes更新・合法手処理）も同予算内に入る。

実運用目標:

- `NN forward + action選択`: 平均 `<= 1.0ms`, P95 `<= 2.0ms`
- `1ターン全体`: 平均 `<= 12-15ms`（安全マージン 5-8ms）

最低ライン:

- `NN` が `>5ms/turn` のモデルは非現実的（他処理の余裕が不足）

### 4.4 Python/C++ の扱い

- 提出版は `C++` 実装前提（Python runtime 依存は不可）。
- 推論器は単一ソース内で完結させる。
- 高速化の基本:
  - `CNN+巨大FC` を避け、`fully-conv` または小型 MLP にする
  - `int8` 重み + `int32` 積和 + requant
  - バッファ再利用（毎ターン malloc しない）
  - 特徴量を全再計算せず、可能なものは差分更新
  - logits は合法手だけ評価する最適化も検討

## 5. モデル候補（研究用と提出用）

以下の「研究用モデル」は精度上限の把握に使い、提出は「提出用モデル」に落とす。

| 区分 | モデル案 | 構造 | パラメータ数 | 速度目安 | 提出可否 |
|---|---|---|---:|---:|---|
| 研究用 | `Teacher-XL` | `cnn_fc` 大型 (`64-128-128 + 大FC`) | `10.8M`（実測） | `~2.3ms`（PyTorch CPU実測） | 不可（容量超過） |
| 研究用 | `Teacher-XXL` | attention/FiLM 追加大型 | `>15M` | `~3.5-5ms` | 不可（容量超過） |
| 提出用 | `Student-S` | fully-conv 小型 (`C=24`, hidden 32) | `~30k` | `<=0.5-1.0ms`（C++想定） | 可 |
| 提出用 | `Student-M` | fully-conv 中型 (`hidden 48-64`) | `~80k-120k` | `<=1.0-1.8ms`（C++想定） | 可（推奨） |
| 提出用 | `Student-L` | fully-conv + 軽量FiLM/attention | `~150k-220k` | `<=1.8-3.0ms`（C++想定） | 条件付き可 |

容量試算（`int8 + base64` のラフ見積り）:

- `Student-S (30k)` -> 重み文字列 `~40-50KiB`
- `Student-M (100k)` -> 重み文字列 `~130-160KiB`
- `Student-L (200k)` -> 重み文字列 `~260-320KiB`

実用結論:

- 最終提出は `Student-M` を主軸にするのが安全。
- `Student-L` はスコア改善が明確な場合のみ採用。

## 6. 実験計画（制約考慮で再設計）

### Phase 0: 特徴量の絞り込み（制約無視）

目的:

- どのチャネルが効くかを確定する（P0/P1 の採否）。

実施:

1. `Teacher-XL` で P0 のみを学習。
2. P1 候補（`A_enemy_e`, `enemy_territory_e`, `enemy_atk_territory`, `A_top`）を1個ずつ追加してアブレーション。
3. 上位 5-8 個の有効特徴に絞る。

### Phase 1: 教師モデルの上限性能を作る（制約無視）

目的:

- 蒸留先のターゲットを作る。

実施:

1. `Teacher-XL`（必要なら XXL）を PPO で収束まで学習。
2. 同一 eval seed 帯でベスト checkpoint を固定。
3. 行動分布（logits）と value を蒸留用データとして保存。

### Phase 2: 提出可能 student へ蒸留

目的:

- 512KiB / 2sec を満たす候補を作る。

実施:

1. `Student-S/M/L` を用意（`30k`, `100k`, `180k` 目安）。
2. 蒸留損失:
   - `L = alpha * CE(a_teacher, a_student) + beta * KL(pi_teacher || pi_student) + gamma * MSE(V_teacher, V_student)`
3. 蒸留後に短い PPO fine-tune（低学習率）を実施。
4. int8 PTQ を実施し、精度劣化が大きければ QAT へ移行。

### Phase 3: 提出版統合（C++）

目的:

- 容量・速度・スコアを同時達成する。

実施:

1. 重みを `int8` へエクスポートし base64 で埋め込み。
2. C++ 推論器で同一入出力を再現（Python と一致確認）。
3. `tools/tester` 上で以下を自動検証:
   - ソースサイズ
   - 1ターン時間（平均/P95）
   - スコア（100/1000 seed）

### Phase 4: 最終選定

採用基準（例）:

1. ソースサイズ `< 500KiB`（バッファ込みで安全側）
2. 1ターン時間 平均 `< 15ms`, P95 `< 20ms`
3. スコアが teacher の `>= 85-90%` を維持

最終方針:

- まず「制約超えでも強い teacher」から知見を取り、
- 最後に「蒸留 + 量子化 + 特徴量選別」で提出制約へ落とし込む。
