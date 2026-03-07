//! Async patterns for EmbedSwarm
//!
//! Adapted from Niodoo-TCS advanced async patterns.
//! Provides priority queues and work-stealing for Swarm concurrency.

use super::types::{Node, NodeClass};
use anyhow::{anyhow, Result};
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Task priority levels for Swarm processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Critical = 0, // Immediate anomaly processing
    High = 1,     // Axiom and hypothesis processing
    Normal = 2,   // Regular observation processing
    Low = 3,      // Background processing
}

impl TaskPriority {
    /// Get priority as numeric value (lower is higher priority)
    pub fn value(&self) -> u8 {
        match self {
            TaskPriority::Critical => 0,
            TaskPriority::High => 1,
            TaskPriority::Normal => 2,
            TaskPriority::Low => 3,
        }
    }

    /// Get time limit for this priority level
    pub fn time_limit(&self) -> Duration {
        match self {
            TaskPriority::Critical => Duration::from_millis(100),
            TaskPriority::High => Duration::from_millis(500),
            TaskPriority::Normal => Duration::from_secs(2),
            TaskPriority::Low => Duration::from_secs(10),
        }
    }

    /// Determine priority from Node class
    pub fn from_node_class(class: &NodeClass) -> Self {
        match class {
            NodeClass::Anomaly => TaskPriority::Critical,
            NodeClass::Axiom => TaskPriority::High,
            NodeClass::Hypothesis => TaskPriority::High,
            NodeClass::Directive => TaskPriority::Normal,
            NodeClass::Observation => TaskPriority::Normal,
        }
    }
}

/// Swarm processing task
#[derive(Debug, Clone)]
pub struct SwarmTask {
    pub id: String,
    pub priority: TaskPriority,
    pub node: Node,
    pub created_at: Instant,
    pub deadline: Option<Instant>,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl SwarmTask {
    /// Create a new Swarm task from a Node
    pub fn new(node: Node) -> Self {
        let priority = TaskPriority::from_node_class(&node.class);
        let created_at = Instant::now();
        let deadline = Some(created_at + priority.time_limit());

        Self {
            id: node.id.clone(),
            priority,
            node,
            created_at,
            deadline,
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Check if task is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Instant::now() > deadline
        } else {
            false
        }
    }

    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Get age of task in milliseconds
    pub fn age_ms(&self) -> u64 {
        self.created_at.elapsed().as_millis() as u64
    }
}

impl PartialEq for SwarmTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.created_at == other.created_at
    }
}

impl Eq for SwarmTask {}

impl PartialOrd for SwarmTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SwarmTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority (lower numeric value) comes first
        let priority_cmp = self.priority.value().cmp(&other.priority.value());
        if priority_cmp != std::cmp::Ordering::Equal {
            return priority_cmp.reverse(); // Reverse because lower priority value = higher priority
        }

        // For same priority, older tasks come first
        self.created_at.cmp(&other.created_at)
    }
}

/// Priority queue for Swarm tasks
pub struct PriorityTaskQueue {
    queue: BinaryHeap<SwarmTask>,
    max_size: usize,
    dropped_tasks: u64,
}

impl PriorityTaskQueue {
    /// Create a new priority task queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: BinaryHeap::new(),
            max_size,
            dropped_tasks: 0,
        }
    }

    /// Add task to queue
    pub fn push(&mut self, task: SwarmTask) -> Result<()> {
        // Drop lowest priority task if queue is full
        if self.queue.len() >= self.max_size {
            if let Some(dropped) = self.queue.pop() {
                self.dropped_tasks += 1;
                warn!("Dropped low-priority task: {}", dropped.id);
            }
        }
        self.queue.push(task);
        Ok(())
    }

    /// Get highest priority task without removing it
    pub fn peek(&self) -> Option<&SwarmTask> {
        self.queue.peek()
    }

    /// Remove and return highest priority task
    pub fn pop(&mut self) -> Option<SwarmTask> {
        self.queue.pop()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get number of dropped tasks
    pub fn dropped_tasks(&self) -> u64 {
        self.dropped_tasks
    }

    /// Clear all tasks
    pub fn clear(&mut self) {
        self.queue.clear();
        self.dropped_tasks = 0;
    }

    /// Get tasks by priority level
    pub fn get_tasks_by_priority(&self, priority: TaskPriority) -> Vec<&SwarmTask> {
        self.queue
            .iter()
            .filter(|task| task.priority == priority)
            .collect()
    }

    /// Get overdue tasks
    pub fn get_overdue_tasks(&self) -> Vec<&SwarmTask> {
        self.queue.iter().filter(|task| task.is_overdue()).collect()
    }
}

/// Simplified work-stealing scheduler for Swarm processing
pub struct WorkStealingScheduler {
    worker_pools: HashMap<String, Arc<WorkerPool>>,
    global_queue: Arc<Mutex<PriorityTaskQueue>>,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub async fn new(num_workers_per_pool: usize) -> Result<Self> {
        info!("Initializing work-stealing scheduler");

        let global_queue = Arc::new(Mutex::new(PriorityTaskQueue::new(10000)));

        // Create worker pools for different processing types
        let mut worker_pools = HashMap::new();
        let pools = vec![
            ("embed_processing", num_workers_per_pool),
            ("function_processing", num_workers_per_pool),
            ("governor_processing", num_workers_per_pool / 2),
        ];

        for (pool_name, num_workers) in pools {
            let worker_pool = Arc::new(
                WorkerPool::new(pool_name.to_string(), num_workers, global_queue.clone()).await?,
            );
            worker_pools.insert(pool_name.to_string(), worker_pool);
        }

        info!(
            "Work-stealing scheduler initialized with {} worker pools",
            worker_pools.len()
        );
        Ok(Self {
            worker_pools,
            global_queue,
        })
    }

    /// Submit task to scheduler
    pub async fn submit_task(&self, task: SwarmTask, preferred_pool: Option<&str>) -> Result<()> {
        // Choose appropriate worker pool
        let target_pool = if let Some(preferred) = preferred_pool {
            self.worker_pools
                .get(preferred)
                .ok_or_else(|| anyhow!("Worker pool not found: {}", preferred))?
        } else {
            // Simple round-robin selection
            let pool_names: Vec<&String> = self.worker_pools.keys().collect();
            if pool_names.is_empty() {
                return Err(anyhow!("No worker pools available"));
            }
            let index = task.id.len() % pool_names.len();
            self.worker_pools.get(pool_names[index]).unwrap()
        };

        // Submit to worker pool
        target_pool.submit_task(task).await
    }

    /// Get scheduler statistics
    pub async fn get_stats(&self) -> Result<SchedulerStats> {
        let mut pool_stats = HashMap::new();
        for (pool_name, pool) in &self.worker_pools {
            pool_stats.insert(pool_name.clone(), pool.get_stats().await?);
        }

        let global_queue_stats = {
            let queue = self.global_queue.lock().await;
            QueueStats {
                queued_tasks: queue.len(),
                dropped_tasks: queue.dropped_tasks(),
            }
        };

        Ok(SchedulerStats {
            pool_stats,
            global_queue_stats,
        })
    }

    /// Shutdown scheduler gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down work-stealing scheduler");

        // Shutdown all worker pools
        for (pool_name, pool) in &self.worker_pools {
            info!("Shutting down worker pool: {}", pool_name);
            pool.shutdown().await?;
        }

        info!("Work-stealing scheduler shutdown complete");
        Ok(())
    }
}

/// Worker pool for processing Swarm tasks
pub struct WorkerPool {
    name: String,
    workers: Vec<tokio::task::JoinHandle<()>>,
    task_queue: Arc<Mutex<PriorityTaskQueue>>,
    global_queue: Arc<Mutex<PriorityTaskQueue>>,
    stats: Arc<Mutex<PoolStats>>,
}

impl WorkerPool {
    /// Create a new worker pool
    pub async fn new(
        name: String,
        num_workers: usize,
        global_queue: Arc<Mutex<PriorityTaskQueue>>,
    ) -> Result<Self> {
        let task_queue = Arc::new(Mutex::new(PriorityTaskQueue::new(1000)));
        let stats = Arc::new(Mutex::new(PoolStats::default()));

        let mut workers = Vec::new();
        for i in 0..num_workers {
            let worker_name = format!("{}_worker_{}", name, i);
            let task_queue_clone = task_queue.clone();
            let global_queue_clone = global_queue.clone();
            let stats_clone = stats.clone();
            let pool_name = name.clone();

            let worker_handle = tokio::spawn(async move {
                Self::worker_loop(
                    worker_name,
                    task_queue_clone,
                    global_queue_clone,
                    stats_clone,
                    pool_name,
                )
                .await;
            });

            workers.push(worker_handle);
        }

        info!(
            "Worker pool '{}' created with {} workers",
            name, num_workers
        );
        Ok(Self {
            name,
            workers,
            task_queue,
            global_queue,
            stats,
        })
    }

    /// Submit task to worker pool
    pub async fn submit_task(&self, task: SwarmTask) -> Result<()> {
        let mut queue = self.task_queue.lock().await;
        queue.push(task)?;
        Ok(())
    }

    /// Worker main loop
    async fn worker_loop(
        worker_name: String,
        task_queue: Arc<Mutex<PriorityTaskQueue>>,
        global_queue: Arc<Mutex<PriorityTaskQueue>>,
        stats: Arc<Mutex<PoolStats>>,
        pool_name: String,
    ) {
        info!("Worker {} started", worker_name);

        loop {
            // Try to get task from local queue first
            let mut task = {
                let mut queue = task_queue.lock().await;
                queue.pop()
            };

            // If no local tasks, try work stealing from global queue
            if task.is_none() {
                task = Self::steal_work(&global_queue, &pool_name).await;
            }

            if let Some(mut task) = task {
                let task_id = task.id.clone();

                // Update stats
                {
                    let mut pool_stats = stats.lock().await;
                    pool_stats.tasks_processed += 1;
                    pool_stats.total_processing_time += task.age_ms();
                }

                // Process the task (simulated)
                let result = Self::process_task(&task).await;
                match result {
                    Ok(_) => {
                        debug!("Task {} completed successfully", task_id);
                    }
                    Err(e) => {
                        warn!("Task {} failed: {}", task_id, e);

                        // Handle retry logic
                        if task.can_retry() {
                            task.increment_retry();

                            // Re-queue task with backoff
                            tokio::time::sleep(Duration::from_millis(
                                100 * task.retry_count as u64,
                            ))
                            .await;

                            let result = {
                                let mut queue = task_queue.lock().await;
                                queue.push(task)
                            };
                            if let Err(retry_err) = result {
                                tracing::error!(
                                    "Failed to re-queue task {}: {}",
                                    task_id,
                                    retry_err
                                );
                            }
                        } else {
                            tracing::error!("Task {} exceeded max retries, dropping", task_id);

                            // Update failure stats
                            let mut pool_stats = stats.lock().await;
                            pool_stats.tasks_failed += 1;
                        }
                    }
                }
            } else {
                // No tasks available, sleep briefly
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    /// Steal work from global queue
    async fn steal_work(
        global_queue: &Arc<Mutex<PriorityTaskQueue>>,
        current_pool: &str,
    ) -> Option<SwarmTask> {
        // Try to steal from global queue
        let mut global = global_queue.lock().await;
        if let Some(task) = global.pop() {
            debug!(
                "Worker from {} stole task {} from global queue",
                current_pool, task.id
            );
            return Some(task);
        }
        None
    }

    /// Process a Swarm task (placeholder)
    async fn process_task(task: &SwarmTask) -> Result<()> {
        // Simulate processing time based on priority
        let processing_time = match task.priority {
            TaskPriority::Critical => Duration::from_millis(50),
            TaskPriority::High => Duration::from_millis(100),
            TaskPriority::Normal => Duration::from_millis(200),
            TaskPriority::Low => Duration::from_millis(500),
        };

        tokio::time::sleep(processing_time).await;

        // Simulate occasional failures for retry testing
        if task.retry_count == 0 && task.id.ends_with("fail") {
            return Err(anyhow!("Simulated processing failure"));
        }

        Ok(())
    }

    /// Get worker pool statistics
    pub async fn get_stats(&self) -> Result<PoolStats> {
        Ok(self.stats.lock().await.clone())
    }

    /// Shutdown worker pool
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down worker pool: {}", self.name);
        // In a real implementation, we'd send shutdown signals to workers
        // For now, we'll just wait for them to finish naturally
        info!("Worker pool {} shutdown complete", self.name);
        Ok(())
    }
}

/// Statistics structures
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub tasks_processed: u64,
    pub tasks_failed: u64,
    pub total_processing_time: u64,
    pub average_processing_time: f64,
}

#[derive(Debug, Clone)]
pub struct QueueStats {
    pub queued_tasks: usize,
    pub dropped_tasks: u64,
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub pool_stats: HashMap<String, PoolStats>,
    pub global_queue_stats: QueueStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concourse::types::NodeClass;

    #[test]
    fn test_task_priority_ordering() {
        let node_critical = crate::concourse::types::Node::new(
            "critical".to_string(),
            NodeClass::Anomaly,
            "test".to_string(),
        );
        let task_critical = SwarmTask::new(node_critical);

        let node_normal = crate::concourse::types::Node::new(
            "normal".to_string(),
            NodeClass::Observation,
            "test".to_string(),
        );
        let task_normal = SwarmTask::new(node_normal);

        // Critical should have higher priority than normal
        assert!(task_critical > task_normal);
    }

    #[test]
    fn test_priority_queue() {
        let mut queue = PriorityTaskQueue::new(3);

        let node_high =
            crate::concourse::types::Node::new("high".to_string(), NodeClass::Axiom, "test".to_string());
        let task_high = SwarmTask::new(node_high);

        let node_low = crate::concourse::types::Node::new(
            "low".to_string(),
            NodeClass::Observation,
            "test".to_string(),
        );
        let task_low = SwarmTask::new(node_low);

        // Add tasks
        queue.push(task_high.clone()).unwrap();
        queue.push(task_low.clone()).unwrap();

        // High priority task should come out first
        let popped = queue.pop().unwrap();
        assert_eq!(popped.priority, TaskPriority::High);
        assert_eq!(popped.id, "high");
    }

    #[tokio::test]
    async fn test_worker_pool() {
        let global_queue = Arc::new(Mutex::new(PriorityTaskQueue::new(100)));
        let worker_pool = WorkerPool::new("test_pool".to_string(), 2, global_queue)
            .await
            .unwrap();

        let node =
            crate::concourse::types::Node::new("test".to_string(), NodeClass::Axiom, "test".to_string());
        let task = SwarmTask::new(node);

        let result = worker_pool.submit_task(task).await;
        assert!(result.is_ok());
    }
}
