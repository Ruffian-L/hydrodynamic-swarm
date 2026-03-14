## 2026-03-14 - Prevent Host-Device Sync Overhead with Tensor::index_select
**Learning:** In Candle, manual iteration over a tensor using `Tensor::get()`, `unsqueeze()`, and `Tensor::cat()` causes repeated host-device synchronization and intermediate tensor allocations inside a loop, resulting in severe performance overhead.
**Action:** Always prefer vectorized operations like `Tensor::index_select` over manual iteration when extracting batches or subsets of a tensor. This keeps operations purely on the device and avoids the synchronization penalty.
