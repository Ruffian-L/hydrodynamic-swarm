//! Shared utilities for the EmbedSwarm system

use std::time::{SystemTime, UNIX_EPOCH};

/// Generate a unique timestamp-based ID
pub fn generate_id(prefix: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{}_{}", prefix, timestamp)
}

/// Truncate text to maximum number of words
pub fn truncate_words(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let truncated: Vec<&str> = words.into_iter().take(max_words).collect();
    truncated.join(" ")
}

/// Calculate simple semantic similarity between texts (placeholder)
pub fn semantic_similarity(text_a: &str, text_b: &str) -> f64 {
    let a_words: std::collections::HashSet<&str> = text_a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = text_b.split_whitespace().collect();

    let intersection: std::collections::HashSet<&str> =
        a_words.intersection(&b_words).cloned().collect();
    let union: std::collections::HashSet<&str> = a_words.union(&b_words).cloned().collect();

    if union.is_empty() {
        0.0
    } else {
        intersection.len() as f64 / union.len() as f64
    }
}

/// Exponential moving average for time series data
pub struct ExponentialMovingAverage {
    alpha: f64,
    value: Option<f64>,
}

impl ExponentialMovingAverage {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, value: None }
    }

    pub fn update(&mut self, new_value: f64) -> f64 {
        match self.value {
            Some(current) => {
                let smoothed = self.alpha * new_value + (1.0 - self.alpha) * current;
                self.value = Some(smoothed);
                smoothed
            }
            None => {
                self.value = Some(new_value);
                new_value
            }
        }
    }

    pub fn current(&self) -> Option<f64> {
        self.value
    }
}

/// Rate limiter for controlling Flux intake
pub struct RateLimiter {
    capacity: usize,
    tokens: usize,
    last_refill: std::time::Instant,
    refill_rate: std::time::Duration,
}

impl RateLimiter {
    pub fn new(capacity: usize, refill_rate: std::time::Duration) -> Self {
        Self {
            capacity,
            tokens: capacity,
            last_refill: std::time::Instant::now(),
            refill_rate,
        }
    }

    pub fn try_acquire(&mut self) -> bool {
        self.refill();

        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill);

        let refills = (elapsed.as_nanos() / self.refill_rate.as_nanos()) as usize;
        if refills > 0 {
            self.tokens = std::cmp::min(self.capacity, self.tokens + refills);
            self.last_refill = now;
        }
    }
}

/// Circular buffer for tracking recent values
pub struct CircularBuffer<T: Clone> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    index: usize,
    size: usize,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            capacity,
            index: 0,
            size: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.buffer[self.index] = Some(value);
        self.index = (self.index + 1) % self.capacity;
        self.size = std::cmp::min(self.size + 1, self.capacity);
    }

    pub fn get(&self, offset: usize) -> Option<&T> {
        if offset >= self.size {
            return None;
        }

        let mut idx = self.index as isize - offset as isize - 1;
        if idx < 0 {
            idx += self.capacity as isize;
        }

        self.buffer[idx as usize].as_ref()
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn iter(&self) -> CircularBufferIterator<'_, T> {
        CircularBufferIterator {
            buffer: self,
            position: 0,
        }
    }
}

pub struct CircularBufferIterator<'a, T: Clone> {
    buffer: &'a CircularBuffer<T>,
    position: usize,
}

impl<'a, T: Clone> Iterator for CircularBufferIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buffer.get(self.position);
        self.position += 1;
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_words() {
        assert_eq!(truncate_words("hello world", 1), "hello");
        assert_eq!(truncate_words("hello world", 5), "hello world");
        assert_eq!(truncate_words("", 5), "");
    }

    #[test]
    fn test_semantic_similarity() {
        assert_eq!(semantic_similarity("hello world", "hello world"), 1.0);
        assert_eq!(semantic_similarity("hello", "world"), 0.0);
        assert!(semantic_similarity("hello world", "hello") > 0.0);
    }

    #[test]
    fn test_ema() {
        let mut ema = ExponentialMovingAverage::new(0.5);
        assert_eq!(ema.update(10.0), 10.0);
        assert_eq!(ema.update(20.0), 15.0);
        assert_eq!(ema.update(30.0), 22.5);
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.get(0), Some(&3));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&1));

        buffer.push(4);
        assert_eq!(buffer.get(0), Some(&4));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(2), Some(&2));
        assert_eq!(buffer.get(3), None);
    }
}
