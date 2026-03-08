
## 2024-03-08 - Use partial sort for Top-K extraction
**Learning:** Using `sort_by` to extract the Top-K items from a large array performs a full `O(N log N)` sort, which can be a severe bottleneck in high-throughput areas like field queries or memory updates.
**Action:** When extracting the Top-K nearest neighbors or elements, use the partial sort method `select_nth_unstable_by` (which is `O(N)`) combined with `truncate` rather than fully sorting the array.
