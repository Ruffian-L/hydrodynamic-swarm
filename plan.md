1. **Optimize `batch_field_gradient` in `metal_backend` fallback (`src/gpu.rs`)**:
   Currently, the fallback to "serial CPU" uses a loop allocating and synchronizing N individual gradients and then using `Tensor::cat`:
   ```rust
   for i in 0..m {
       let pos_i = positions.get(i)?;
       let grad_i = field.probe_gradient(&pos_i)?.unsqueeze(0)?;
       grads.push(grad_i);
   }
   return Tensor::cat(&grads, 0);
   ```
   Instead of iterating over rows, getting individual tensors, unsqueezing and concatenating them, which is very inefficient for device dispatch and memory allocation, we can replace this with `CpuBackend::new().batch_field_gradient(field, positions)` or just the actual vectorized CPU implementation itself since `CpuBackend` already implements it using broadcasting without loops. Wait, actually, `CpuBackend` has a `batch_field_gradient` method that does vectorized broadcast math correctly! We can just use it.
