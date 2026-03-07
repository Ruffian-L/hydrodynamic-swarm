//! Caching system for EmbedSwarm
//!
//! Provides LRU and TTL caches for embeddings and edge calculations
//! Adapted from Niodoo-TCS advanced caching system.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub created_at: SystemTime,
    pub ttl_seconds: Option<u64>,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(key: String, value: Vec<u8>, ttl_seconds: Option<u64>) -> Self {
        Self {
            key,
            value,
            created_at: SystemTime::now(),
            ttl_seconds,
        }
    }

    /// Check if entry is expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_seconds {
            let now = SystemTime::now();
            let elapsed = now.duration_since(self.created_at).unwrap_or_default();
            elapsed.as_secs() > ttl
        } else {
            false
        }
    }
}

/// LRU cache implementation
pub struct LruCache {
    capacity: usize,
    entries: HashMap<String, CacheEntry>,
    access_order: VecDeque<String>,
}

impl LruCache {
    /// Create a new LRU cache
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    /// Get value from cache (returns cloned entry)
    pub fn get(&mut self, key: &str) -> Option<CacheEntry> {
        let entry_clone = self.entries.get(key).cloned();
        match entry_clone {
            Some(entry) if !entry.is_expired() => {
                // valid entry, update access order
                let pos = self.access_order.iter().position(|k| k == key);
                if let Some(pos) = pos {
                    self.access_order.remove(pos);
                }
                self.access_order.push_back(key.to_string());
                Some(entry)
            }
            _ => {
                // either missing or expired
                self.remove(key);
                None
            }
        }
    }

    /// Put value in cache
    pub fn put(&mut self, key: String, value: Vec<u8>, ttl_seconds: Option<u64>) {
        // Remove expired entries first
        self.cleanup();

        let is_update = self.entries.contains_key(&key);

        // If at capacity and not an update, remove least recently used
        if self.entries.len() >= self.capacity && !is_update {
            if let Some(lru_key) = self.access_order.pop_front() {
                self.entries.remove(&lru_key);
            }
        }

        // If updating, remove old position from access_order
        if is_update {
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        }

        let entry = CacheEntry::new(key.clone(), value, ttl_seconds);
        self.entries.insert(key.clone(), entry);
        self.access_order.push_back(key);
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            Some(entry)
        } else {
            None
        }
    }

    /// Remove expired entries
    pub fn cleanup(&mut self) {
        let mut expired_keys = Vec::new();
        for (key, entry) in &self.entries {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            self.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_size: usize = self.entries.values().map(|e| e.value.len()).sum();
        CacheStats {
            entries: total_entries,
            total_size_bytes: total_size,
            hit_rate: 0.0, // Simplified - no access tracking
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// TTL-based cache
pub struct TtlCache {
    entries: HashMap<String, CacheEntry>,
    default_ttl_seconds: u64,
}

impl TtlCache {
    /// Create a new TTL cache
    pub fn new(default_ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_seconds,
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        use std::collections::hash_map::Entry;
        match self.entries.entry(key.to_string()) {
            Entry::Occupied(occupied) => {
                if occupied.get().is_expired() {
                    occupied.remove();
                    None
                } else {
                    Some(&*occupied.into_mut())
                }
            }
            Entry::Vacant(_) => None,
        }
    }

    /// Put value in cache with TTL
    pub fn put(&mut self, key: String, value: Vec<u8>, ttl_seconds: Option<u64>) {
        let ttl = ttl_seconds.unwrap_or(self.default_ttl_seconds);
        let entry = CacheEntry::new(key.clone(), value, Some(ttl));
        self.entries.insert(key, entry);
    }

    /// Remove expired entries
    pub fn cleanup(&mut self) {
        let mut expired_keys = Vec::new();
        for (key, entry) in &self.entries {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            self.entries.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_size: usize = self.entries.values().map(|e| e.value.len()).sum();
        CacheStats {
            entries: total_entries,
            total_size_bytes: total_size,
            hit_rate: 0.0,
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Simplified cache manager combining LRU and TTL caches
pub struct CacheManager {
    lru_cache: Arc<RwLock<LruCache>>,
    ttl_cache: Arc<RwLock<TtlCache>>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new() -> Self {
        Self {
            lru_cache: Arc::new(RwLock::new(LruCache::new(1000))), // 1000 entries
            ttl_cache: Arc::new(RwLock::new(TtlCache::new(3600))), // 1 hour default TTL
        }
    }

    /// Get cached embedding
    pub fn get_embedding(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let cache_key = Self::generate_cache_key(text, "embedding");

        // Check LRU cache first
        if let Some(entry) = self.lru_cache.write().unwrap().get(&cache_key) {
            let embedding: Vec<f32> = bincode::deserialize(&entry.value)
                .map_err(|e| anyhow!("Failed to deserialize embedding: {}", e))?;
            return Ok(Some(embedding));
        }

        // Check TTL cache
        if let Some(entry) = self.ttl_cache.write().unwrap().get(&cache_key) {
            // Promote to LRU cache
            let embedding: Vec<f32> = bincode::deserialize(&entry.value)
                .map_err(|e| anyhow!("Failed to deserialize embedding: {}", e))?;

            let serialized = bincode::serialize(&embedding)
                .map_err(|e| anyhow!("Failed to serialize embedding: {}", e))?;

            self.lru_cache.write().unwrap().put(
                cache_key.clone(),
                serialized.clone(),
                entry.ttl_seconds,
            );

            return Ok(Some(embedding));
        }

        Ok(None)
    }

    /// Cache embedding result
    pub fn cache_embedding(&self, text: &str, embedding: &[f32]) -> Result<()> {
        let cache_key = Self::generate_cache_key(text, "embedding");
        let serialized = bincode::serialize(embedding)
            .map_err(|e| anyhow!("Failed to serialize embedding: {}", e))?;

        // Store in both caches
        self.lru_cache.write().unwrap().put(
            cache_key.clone(),
            serialized.clone(),
            Some(300), // 5 minute TTL for LRU
        );

        self.ttl_cache.write().unwrap().put(
            cache_key,
            serialized,
            Some(3600), // 1 hour TTL for TTL cache
        );

        Ok(())
    }

    /// Get cached edge relationship
    pub fn get_edge_relationship(&self, text_a: &str, text_b: &str) -> Result<Option<Vec<u8>>> {
        let cache_key = Self::generate_edge_key(text_a, text_b);

        // Check LRU cache
        if let Some(entry) = self.lru_cache.write().unwrap().get(&cache_key) {
            return Ok(Some(entry.value.clone()));
        }

        // Check TTL cache
        if let Some(entry) = self.ttl_cache.write().unwrap().get(&cache_key) {
            // Promote to LRU cache
            self.lru_cache.write().unwrap().put(
                cache_key.clone(),
                entry.value.clone(),
                entry.ttl_seconds,
            );
            return Ok(Some(entry.value.clone()));
        }

        Ok(None)
    }

    /// Cache edge relationship result
    pub fn cache_edge_relationship(&self, text_a: &str, text_b: &str, result: &[u8]) -> Result<()> {
        let cache_key = Self::generate_edge_key(text_a, text_b);

        self.lru_cache.write().unwrap().put(
            cache_key.clone(),
            result.to_vec(),
            Some(600), // 10 minute TTL for edges
        );

        self.ttl_cache.write().unwrap().put(
            cache_key,
            result.to_vec(),
            Some(1800), // 30 minute TTL for edges
        );

        Ok(())
    }

    /// Cleanup expired entries
    pub fn cleanup(&self) -> Result<()> {
        self.lru_cache.write().unwrap().cleanup();
        self.ttl_cache.write().unwrap().cleanup();
        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> Result<HashMap<String, CacheStats>> {
        let mut stats = HashMap::new();
        stats.insert(
            "lru_cache".to_string(),
            self.lru_cache.read().unwrap().stats(),
        );
        stats.insert(
            "ttl_cache".to_string(),
            self.ttl_cache.read().unwrap().stats(),
        );
        Ok(stats)
    }

    /// Clear all caches
    pub fn clear_all(&self) -> Result<()> {
        self.lru_cache.write().unwrap().clear();
        self.ttl_cache.write().unwrap().clear();
        Ok(())
    }

    /// Generate cache key from input
    fn generate_cache_key(input: &str, prefix: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        prefix.hash(&mut hasher);
        input.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Generate cache key for edge relationship
    fn generate_edge_key(text_a: &str, text_b: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        "edge".hash(&mut hasher);
        text_a.hash(&mut hasher);
        text_b.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entries: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);

        // Test basic operations
        cache.put("key1".to_string(), b"value1".to_vec(), None);
        cache.put("key2".to_string(), b"value2".to_vec(), None);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("key1").is_some());

        // Add third item should evict least recently used (key2)
        cache.put("key3".to_string(), b"value3".to_vec(), None);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("key1").is_some()); // key1 accessed, should remain
        assert!(cache.get("key2").is_none()); // key2 not accessed, should be evicted
        assert!(cache.get("key3").is_some()); // Should be added
    }

    #[test]
    fn test_ttl_cache() {
        let mut cache = TtlCache::new(1); // 1 second TTL

        // Add entry
        cache.put("key1".to_string(), b"value1".to_vec(), Some(1));

        // Should be available immediately
        assert!(cache.get("key1").is_some());
    }

    #[test]
    fn test_cache_manager() {
        let cache_manager = CacheManager::new();

        // Test embedding caching
        let embedding = vec![1.0, 2.0, 3.0];
        let result = cache_manager.cache_embedding("test_text", &embedding);
        assert!(result.is_ok());

        let retrieved = cache_manager.get_embedding("test_text").unwrap();
        assert_eq!(retrieved, Some(embedding));

        // Test cache statistics
        let stats = cache_manager.get_stats().unwrap();
        assert!(stats.contains_key("lru_cache"));
        assert!(stats.contains_key("ttl_cache"));
    }
}
