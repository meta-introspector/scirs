# 具体例: scirs2-optimize モジュールの移行

実際にscirs2-optimizeモジュールのSIMD実装を移行する手順を示します。

## 現状分析

### 1. 問題のあるファイル
- `/scirs2-optimize/src/simd_ops.rs` - 完全に置き換える必要あり
- `/scirs2-optimize/src/unconstrained/simd_bfgs.rs` - リファクタリング必要

### 2. 現在の実装例（simd_ops.rs）

```rust
// 現在の実装（削除対象）
use std::arch::x86_64::*;

pub struct SimdConfig {
    pub use_simd: bool,
    pub simd_width: usize,
}

pub fn dot_product_simd(x: &[f64], y: &[f64]) -> f64 {
    unsafe {
        let mut sum = _mm256_setzero_pd();
        // AVX2を使った実装...
    }
}
```

## Step 1: Cargo.tomlの更新

```toml
# scirs2-optimize/Cargo.toml

[dependencies]
scirs2-core = { workspace = true, features = ["simd", "parallel"] }

# これらを削除（またはdev-dependenciesへ）
# [target.'cfg(target_arch = "x86_64")'.dependencies]
# なし（直接arch使用をやめる）
```

## Step 2: simd_ops.rsの完全置き換え

### 新しいsimd_ops.rs
```rust
//! SIMD operations for optimization algorithms
//!
//! This module provides SIMD-accelerated operations by delegating
//! to scirs2-core's unified SIMD system.

use ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities, AutoOptimizer};

/// Check if SIMD is available
pub fn simd_available() -> bool {
    let caps = PlatformCapabilities::detect();
    caps.simd_available
}

/// Optimized dot product
pub fn dot_product<T: SimdUnifiedOps>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
    T::simd_dot(x, y)
}

/// Optimized vector addition: x + alpha * y
pub fn axpy<T: SimdUnifiedOps>(alpha: T, x: &ArrayView1<T>, y: &mut Array1<T>) {
    let scaled_x = T::simd_scalar_mul(x, alpha);
    let sum = T::simd_add(&scaled_x.view(), &y.view());
    y.assign(&sum);
}

/// Optimized vector norm
pub fn norm<T: SimdUnifiedOps>(x: &ArrayView1<T>) -> T {
    T::simd_norm(x)
}

/// Element-wise operations
pub fn elementwise_mul<T: SimdUnifiedOps>(
    x: &ArrayView1<T>, 
    y: &ArrayView1<T>
) -> Array1<T> {
    T::simd_mul(x, y)
}

/// Auto-optimized operations based on problem size
pub struct OptimizedOps {
    optimizer: AutoOptimizer,
}

impl OptimizedOps {
    pub fn new() -> Self {
        Self {
            optimizer: AutoOptimizer::new(),
        }
    }
    
    pub fn should_use_simd(&self, size: usize) -> bool {
        self.optimizer.should_use_simd(size)
    }
}
```

## Step 3: simd_bfgs.rsの更新

### Before:
```rust
use crate::simd_ops::{dot_product_simd, SimdConfig};

impl SimdBFGS {
    fn compute_direction(&self, grad: &[f64], h: &[f64]) -> Vec<f64> {
        if self.config.use_simd {
            // カスタムSIMD実装
            dot_product_simd(grad, h)
        } else {
            // スカラー実装
        }
    }
}
```

### After:
```rust
use ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

pub struct SimdBFGS<T: SimdUnifiedOps> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: SimdUnifiedOps + Float> SimdBFGS<T> {
    fn compute_direction(
        &self, 
        grad: &ArrayView1<T>, 
        h: &ArrayView2<T>
    ) -> Array1<T> {
        // 自動的に最適な実装が選択される
        h.dot(grad)  // ndarrayのdotは内部でBLASを使用
    }
    
    fn update_hessian_approx(
        &mut self,
        s: &ArrayView1<T>,
        y: &ArrayView1<T>,
        h: &mut Array2<T>,
    ) {
        let rho = T::one() / T::simd_dot(y, s);
        let sy = T::simd_mul(s, y);
        
        // BFGS更新式をSIMD演算で実装
        // ...
    }
}
```

## Step 4: 既存テストの維持

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dot_product_consistency() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![5.0, 6.0, 7.0, 8.0];
        
        // 新しい実装でも同じ結果
        let result = dot_product(&x.view(), &y.view());
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    }
}
```

## Step 5: ベンチマークで検証

```rust
// benches/simd_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_dot_product(c: &mut Criterion) {
    let size = 10000;
    let x = Array1::random(size, Uniform::new(-1.0, 1.0));
    let y = Array1::random(size, Uniform::new(-1.0, 1.0));
    
    c.bench_function("dot_product_unified", |b| {
        b.iter(|| {
            dot_product(&x.view(), &y.view())
        })
    });
}
```

## Step 6: 段階的な移行

### Phase 1: 基本関数（1日目）
```bash
# 基本的なSIMD関数を移行
- dot_product
- vector_add
- vector_scale
- norm

# テスト実行
cargo test --features simd
```

### Phase 2: 複雑な最適化アルゴリズム（2-3日目）
```bash
# BFGS, L-BFGS などの実装を更新
- Hessian更新
- Line search
- Trust region

# ベンチマーク実行
cargo bench
```

### Phase 3: クリーンアップ（4日目）
```bash
# 古いコードを削除
rm src/simd_ops_old.rs

# ドキュメント更新
cargo doc --open
```

## トラブルシューティング

### 問題: スライスからArrayViewへの変換

```rust
// 問題のあるコード
pub fn optimize(x: &mut [f64]) {
    let grad = compute_gradient(x);
    let dot = dot_product_simd(x, &grad); // エラー！
}

// 解決策
use ndarray::{ArrayView1, ArrayViewMut1};

pub fn optimize(x: &mut [f64]) {
    let mut x_array = ArrayViewMut1::from_shape(x.len(), x).unwrap();
    let grad = compute_gradient(&x_array.view());
    let dot = f64::simd_dot(&x_array.view(), &grad.view());
}
```

### 問題: パフォーマンスの低下

```rust
// パフォーマンスが重要な場合の対処
pub fn critical_path_operation(data: &ArrayView1<f64>) -> f64 {
    let optimizer = AutoOptimizer::new();
    
    // サイズに基づいて最適な実装を選択
    if data.len() < 64 {
        // 小さいデータはスカラー実装の方が速い
        data.iter().sum()
    } else {
        // 大きいデータはSIMD
        f64::simd_sum(data)
    }
}
```

## 移行の利点

1. **保守性向上**: カスタムSIMDコードの削除
2. **移植性向上**: プラットフォーム依存コードの削除
3. **将来性**: 新しいSIMD命令への自動対応
4. **統一性**: 他のモジュールと同じAPI

## 最終チェック

- [ ] すべてのテストが通る
- [ ] ベンチマークで大きな性能低下がない
- [ ] ドキュメントが更新されている
- [ ] 不要な依存関係が削除されている
- [ ] Clippyの警告がない