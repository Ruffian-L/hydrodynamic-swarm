## 2024-05-24 - Faster Top-K element extraction in Candle
**Learning:** When extracting a subset of elements (like Top-K) from a tensor batch in Candle, using `Tensor::index_select` is significantly faster than mapping indices to `unsqueeze(0)` tensors and using `Tensor::cat`. `index_select` performs the operation natively on the target device and avoids many intermediate heap allocations.
**Action:** Use `Tensor::index_select` with a 1D tensor of indices instead of `Tensor::get().unsqueeze()` and `Tensor::cat()` for gathering elements from batch tensors.
