## 2026-03-10 - Avoid `get().unsqueeze()` + `cat()` for batch row retrieval in Candle
**Learning:** In the Candle ML framework, looping over rows to extract them individually with `Tensor::get().unsqueeze()` and then concatenating them with `Tensor::cat()` is extremely slow, as it causes O(K) intermediate tensor allocations and significant host-device synchronization overhead.
**Action:** Use `Tensor::index_select` with a 1D index tensor instead. This allows the backend to perform the extraction in a single optimized kernel call, minimizing both memory allocations and sync overhead.
