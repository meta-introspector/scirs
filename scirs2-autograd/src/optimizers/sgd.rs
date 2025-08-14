//! SGD optimizer

use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use crate::tensor_ops::gradient_descent_ops::sgd;

use crate::{Context, Float};

/// Gradient descent optimizer
///
///    ```
/// use scirs2_autograd as ag;
/// use ag::prelude::*;
/// use ag::optimizers::SGD;
///
/// type Tensor<'g> = ag::Tensor<'g, f64>;
/// let mut env = ag::VariableEnvironment::new();
/// let opt = SGD::new(0.01);
///
/// env.run(|g| {
///    let p = g.placeholder("p", &[]);
///
///    let mut feeder = ag::Feeder::new();
///    let feed = ag::ndarray::arr0(2.);
///    let feeder = feeder.push(p, feed.view().into_dyn());
///
///    let (params, grads): (&[Tensor], &[Tensor]) = (&[], &[]); // dummy
///    opt.update(params, grads, g, feeder); // do parameter update
/// });
///    ```
pub struct SGD<F> {
    pub alpha: F,
}

impl<F: Float> SGD<F> {
    pub fn new(alpha: F) -> SGD<F> {
        SGD { alpha }
    }
}

impl<F: Float> Optimizer<F> for SGD<F> {
    fn compute_updates<'g, A, B>(
        &self,
        params: &[A],
        grads: &[B],
        ctx: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        let num_params = params.len();
        assert_eq!(num_params, grads.len());
        let mut ret = Vec::with_capacity(num_params);
        for i in 0..num_params {
            ret.push(
                Tensor::builder(ctx)
                    .append_input(params[i].as_ref(), true)
                    .append_input(grads[i].as_ref(), false)
                    .build(sgd::SGDOp { alpha: self.alpha }),
            );
        }
        ret
    }
}
