# AHC061 C++ Core Layout

`ahc061/core` は以下のモジュール単位で分割する。

- `base/`
  - 基本型・定数 (`state.hpp`)
- `game/`
  - ルール、スコア、ケース生成 (`rules.hpp`, `score.hpp`, `generator.hpp`)
- `opponent/`
  - 敵行動モデルと推定器 (`pf.hpp`, `a_softmax_laplace.hpp`, `adf_beta_estimator.hpp`)
- `features/`
  - 観測特徴とレジストリ (`feature_common.hpp`, `feature_registry.hpp`, `features_*.hpp`)
  - Teacher の Bayes 補助は `features_teacher_bayes.hpp` に分離
- `env/`
  - バッチ環境本体 (`env.hpp`)
- `io/`
  - tools 入力読み込み (`tools_input.hpp`)

include はモジュール経路を明示する:

- 例: `#include "ahc061/features/feature_registry.hpp"`
