## 2025-03-09 - Tensor::index_select instead of manual loops for batching
**Learning:** In candle_core, prefer `Tensor::index_select` with a 1D index tensor over looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval. `index_select` minimizes host-device synchronization and intermediate allocations, providing significant performance gains.
**Action:** Replace `Tensor::cat` loops with `Tensor::index_select` where applicable.
