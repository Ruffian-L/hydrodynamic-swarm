## 2024-05-18 - Optimized `probe_gradient_topk` using index_select
**Learning:** In candle_core, prefer `Tensor::index_select` with a 1D index tensor over looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval. `index_select` minimizes host-device synchronization and intermediate allocations, providing significant performance gains.
**Action:** Convert indices via `Tensor::from_vec(indices, (len,), &device)` before selecting, and ensure the source vector is typed as `u32`.
