# Known Issues - SciRS2 Core

The array protocol implementation is currently in active development and has known test failures in the following areas:

- Distributed arrays: `test_distributed_ndarray_creation`, `test_distributed_ndarray_to_array`
- Custom array types: `example_custom_array_type`, `example_distributed_array`
- Gradient computation: `test_gradient_computation_add`, `test_gradient_computation_multiply`, `test_sgd_optimizer`
- Mixed precision: `test_mixed_precision_array`
- Array operations: `test_operations_with_ndarray`
- Serialization: `test_model_serializer`, `test_save_load_checkpoint`
- Training: `test_mse_loss`

These failures are expected as part of the ongoing implementation work and will be addressed in future releases.