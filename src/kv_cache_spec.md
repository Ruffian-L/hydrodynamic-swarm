# Choice-Driven KV Cache

We are abandoning mechanical LRU lobotomies. The engine will manage KV cache retention actively.

## Model Interface Additions
To support Sinks, Slipping Streams, Pins, and the Sandbox, the physical `Model` struct must expose:

1. `pub fn retain_kv(&mut self, keep_indices: &Tensor)`
   - Takes a 1D tensor of `u32` indices representing the absolute positions to keep.
   - Slices the `FullAttention` KV caches (via `index_select` along dim 2).
   - Leaves Mamba / Linear attention states completely untouched (they are constant size O(1) and act as long-term intuition).

2. `pub fn snapshot_kv(&self) -> KvSnapshot`
   - Clones the `Option<(Tensor, Tensor)>` Arcs for all layers. This is perfectly zero-copy memory-wise until mutated!
   - Clones the Mamba `conv_state` and `ssm_state` (which are trivially small).

3. `pub fn restore_kv(&mut self, snap: &KvSnapshot)`
   - Reverts the model's KV state back to the snapshot, instantly wiping the Sandbox preview from history without a trace.
