## 2024-05-15 - [Top-K Gradient Optimization]
**Learning:** Found an opportunity to use `probe_gradient_topk` instead of `probe_gradient` in some places to potentially avoid calculating gradients over all positions when it's not necessary.
**Action:** Investigate the performance impact of using `probe_gradient_topk` with an appropriate `k` in more places where `field_gradient` is used on the CPU.
## 2024-05-15 - [Top-K Partial Sorting in Memory Retrieval]
**Learning:** Found that using full sorts (`sort_by`) for extracting the top-K elements in memory retrievals scales as O(N log N) and becomes a bottleneck when N is large.
**Action:** Replace full sorts with partial sorts (`select_nth_unstable_by`) when finding the top-K nearest neighbors. This reduces the complexity to O(N) and prevents unnecessary sorting of the remaining N-K elements.
