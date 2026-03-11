# Bolt Journal


## 2025-03-11 - [Use index_select for batch row retrieval in candle_core]
**Learning:** In `candle_core`, gathering rows from a large tensor by looping over indices with `Tensor::get().unsqueeze()` and joining with `Tensor::cat()` creates multiple intermediate allocations and host-device synchronization calls, leading to significant performance overhead.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor for batch retrieval of rows. This minimizes host-device sync and executes as a highly optimized O(1) batch operation on the device.
