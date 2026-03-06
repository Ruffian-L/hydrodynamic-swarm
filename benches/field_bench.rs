use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simulate_bottleneck(dist_sq: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = dist_sq.len();
    let mut indices: Vec<(usize, f32)> =
        dist_sq.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    let k = k.min(n);
    if k == 0 || indices.is_empty() {
        return Vec::new();
    }
    indices.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);
    indices
}

fn simulate_optimized(dist_sq: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = dist_sq.len();
    let k = k.min(n);
    if k == 0 || dist_sq.is_empty() {
        return Vec::new();
    }

    // Using a max-heap to keep track of the smallest k elements
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    #[derive(PartialEq)]
    struct OrderedF32(f32);

    impl Eq for OrderedF32 {}

    impl PartialOrd for OrderedF32 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl Ord for OrderedF32 {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::with_capacity(k);

    for (i, &d) in dist_sq.iter().enumerate() {
        if heap.len() < k {
            heap.push((OrderedF32(d), i));
        } else if let Some(max_val) = heap.peek() {
            if d < max_val.0.0 {
                heap.pop();
                heap.push((OrderedF32(d), i));
            }
        }
    }

    heap.into_iter().map(|(d, i)| (i, d.0)).collect()
}

fn criterion_benchmark(c: &mut Criterion) {
    let n = 128_000; // Simulated vocabulary size

    // Create random distances
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(42);
    let dist_sq: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();

    c.bench_function("bottleneck_k5", |b| b.iter(|| {
        std::hint::black_box(simulate_bottleneck(&dist_sq, 5));
    }));

    c.bench_function("optimized_k5", |b| b.iter(|| {
        std::hint::black_box(simulate_optimized(&dist_sq, 5));
    }));

    c.bench_function("bottleneck_k2048", |b| b.iter(|| {
        std::hint::black_box(simulate_bottleneck(&dist_sq, 2048));
    }));

    c.bench_function("optimized_k2048", |b| b.iter(|| {
        std::hint::black_box(simulate_optimized(&dist_sq, 2048));
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
