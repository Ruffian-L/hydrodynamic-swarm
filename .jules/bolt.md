## 2025-02-14 - Optimize Tensor Row Extraction with `index_select`
**Learning:** In `candle-core`, iterating over a vector of indices, extracting rows via `.get()`, `.unsqueeze(0)`, and concatenating them via `Tensor::cat()` creates O(K) intermediate tensor allocations and significant host-device synchronization overhead.
**Action:** Always prefer `Tensor::index_select` with a 1D tensor of `u32` indices for batched row extraction. This offloads the entire gather operation to the backend natively and reduces execution time dramatically.
