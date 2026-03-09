## 2025-03-08 - Use partial sort for Top-K
**Learning:** Extracting Top-K elements from a collection using a full sort (`sort_by`) is O(N log N) and can be a significant performance bottleneck when the dataset is large.
**Action:** Replace `sort_by` followed by `take` or `truncate` with `select_nth_unstable_by`. It provides partial sorting in O(N) time, keeping the top K elements at the front of the collection. Always handle edge cases like `k == 0` appropriately.
