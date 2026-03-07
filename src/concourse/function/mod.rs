//! Function Gemmas: Connective tissue that draws relational edges between Nodes
//!
//! 5-instance worker pool that analyzes proximity and determines relational gravity
//! using the strict lexicon of 7 Relational Edges.

use crate::concourse::async_patterns::{PriorityTaskQueue, SwarmTask};
use crate::concourse::physics::CognitiveState;
use crate::concourse::types::{Edge, FluxTuple, Node, NodeClass, RelationalEdge};
use crate::concourse::{SwarmError, SwarmResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

#[cfg(feature = "with-candle")]
pub mod instruct_gemma;
#[cfg(feature = "with-candle")]
use instruct_gemma::classify_edge_llm;

/// Graph representation for proximity calculations
#[derive(Debug, Default)]
pub struct ActiveGraph {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<Edge>,
}

impl ActiveGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Find k-nearest neighbors to a node (simplified placeholder)
    pub fn find_nearest_neighbors(&self, node_id: &str, k: usize) -> Vec<Node> {
        let mut distances: Vec<(f64, Node)> = Vec::new();

        if let Some(target_node) = self.nodes.get(node_id) {
            for (_, node) in &self.nodes {
                if node.id != node_id {
                    // Simplified distance calculation
                    let distance = calculate_semantic_distance(
                        &target_node.semantic_hash,
                        &node.semantic_hash,
                    );
                    distances.push((distance, node.clone()));
                }
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances
            .into_iter()
            .take(k)
            .map(|(_, node)| node)
            .collect()
    }
}

/// Function Gemma worker instance
#[derive(Clone)]
pub struct FunctionGemmaWorker {
    _worker_id: usize,
    tx: mpsc::Sender<FluxTuple>,
}

impl FunctionGemmaWorker {
    pub fn new(worker_id: usize, tx: mpsc::Sender<FluxTuple>) -> Self {
        Self {
            _worker_id: worker_id,
            tx,
        }
    }

    /// Analyze relationship between new node and existing graph
    pub async fn analyze_relationships(
        &self,
        new_node: Node,
        graph: Arc<RwLock<ActiveGraph>>,
    ) -> SwarmResult<()> {
        let neighbors = {
            let graph_read = graph.read().await;
            graph_read.find_nearest_neighbors(&new_node.id, 5)
        };

        for neighbor in neighbors {
            let edge = self.determine_relationship(&new_node, &neighbor);

            self.tx
                .send(edge.clone().into())
                .await
                .map_err(|e| SwarmError::Concurrency(format!("Failed to send edge: {}", e)))?;
        }

        Ok(())
    }

    /// Determine relationship between two nodes using real LLM or heuristic fallback
    fn determine_relationship(&self, node_a: &Node, node_b: &Node) -> Edge {
        let text_a = &node_a.semantic_hash;
        let text_b = &node_b.semantic_hash;

        // Try LLM classification first (no-op if model unavailable)
        #[cfg(feature = "with-candle")]
        if let Some(edge) = classify_edge_llm(text_a, text_b) {
            return Edge::new(node_a.id.clone(), edge, node_b.id.clone());
        }

        // Heuristic fallback (always available)
        let edge = if text_a.contains("ERROR") && text_b.contains("SOLUTION") {
            RelationalEdge::Synthesizes
        } else if text_a.contains("BASE") && text_b.contains("EXTENSION") {
            RelationalEdge::Scaffolds
        } else if text_a.contains("CONFLICT") && text_b.contains("CONFLICT") {
            RelationalEdge::Contradicts
        } else if text_a.contains("TRIGGER") && text_b.contains("ACTION") {
            RelationalEdge::Actuates
        } else if text_a.contains("CONTAINER") && text_b.contains("CONTENT") {
            RelationalEdge::Encapsulates
        } else if text_a.contains("PATTERN") && text_b.contains("PATTERN") {
            RelationalEdge::IsIsomorphicTo
        } else {
            RelationalEdge::Actuates
        };

        Edge::new(node_a.id.clone(), edge, node_b.id.clone())
    }
}

/// Manager for the 5-instance Function Gemma worker pool with priority queuing
#[derive(Clone)]
pub struct FunctionManager {
    task_queue: Arc<RwLock<PriorityTaskQueue>>,
    worker_handles: Arc<RwLock<Vec<JoinHandle<()>>>>,
    graph: Arc<RwLock<ActiveGraph>>,
    cognitive_state: Arc<RwLock<CognitiveState>>,
    edge_tx: mpsc::Sender<FluxTuple>,
}

impl FunctionManager {
    pub fn new(
        graph: Arc<RwLock<ActiveGraph>>,
        cognitive_state: Arc<RwLock<CognitiveState>>,
        edge_tx: mpsc::Sender<FluxTuple>,
    ) -> Self {
        let task_queue = Arc::new(RwLock::new(PriorityTaskQueue::new(1000)));
        let worker_handles = Arc::new(RwLock::new(Vec::new()));

        let manager = Self {
            task_queue: task_queue.clone(),
            worker_handles: worker_handles.clone(),
            graph: graph.clone(),
            cognitive_state: cognitive_state.clone(),
            edge_tx: edge_tx.clone(),
        };

        // Spawn worker tasks
        let manager_clone = manager.clone();
        tokio::spawn(async move {
            manager_clone.start_workers(5).await;
        });

        manager
    }

    /// Start worker tasks
    async fn start_workers(&self, num_workers: usize) {
        let mut handles = Vec::new();

        for worker_id in 0..num_workers {
            let task_queue = self.task_queue.clone();
            let graph = self.graph.clone();
            let cognitive_state = self.cognitive_state.clone();
            let edge_tx = self.edge_tx.clone();

            let handle = tokio::spawn(async move {
                Self::worker_loop(worker_id, task_queue, graph, cognitive_state, edge_tx).await;
            });

            handles.push(handle);
        }

        *self.worker_handles.write().await = handles;
        info!(
            "Started {} function workers with priority queuing",
            num_workers
        );
    }

    /// Worker main loop
    async fn worker_loop(
        worker_id: usize,
        task_queue: Arc<RwLock<PriorityTaskQueue>>,
        graph: Arc<RwLock<ActiveGraph>>,
        cognitive_state: Arc<RwLock<CognitiveState>>,
        edge_tx: mpsc::Sender<FluxTuple>,
    ) {
        debug!("Function worker {} started", worker_id);
        let worker = FunctionGemmaWorker::new(worker_id, edge_tx);

        loop {
            // Get highest priority task
            let task = {
                let mut queue = task_queue.write().await;
                queue.pop()
            };

            if let Some(task) = task {
                debug!("Worker {} processing task {}", worker_id, task.id);

                // Add node to graph
                {
                    let mut graph_write = graph.write().await;
                    graph_write.add_node(task.node.clone());
                }

                // Update cognitive state based on node class
                {
                    let mut state_write = cognitive_state.write().await;
                    match task.node.class {
                        NodeClass::Anomaly => state_write.omega += 0.1,
                        NodeClass::Axiom => state_write.k_coupling += 0.05,
                        _ => {}
                    }
                }

                // Analyze relationships (clone node to avoid moving task)
                let node = task.node.clone();
                if let Err(e) = worker.analyze_relationships(node, graph.clone()).await {
                    error!(
                        "Worker {} failed to analyze relationships: {}",
                        worker_id, e
                    );

                    // Retry logic
                    if task.can_retry() {
                        let mut new_task = task;
                        new_task.increment_retry();
                        debug!(
                            "Re-queueing task {} for retry {}",
                            new_task.id, new_task.retry_count
                        );

                        let mut queue = task_queue.write().await;
                        if let Err(retry_err) = queue.push(new_task) {
                            error!("Failed to re-queue task: {}", retry_err);
                        }
                    }
                }
            } else {
                // No tasks, sleep briefly
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }
    }

    /// Distribute node analysis across worker pool via priority queue
    pub async fn process_node(&self, node: Node) -> SwarmResult<()> {
        let task = SwarmTask::new(node);
        debug!(
            "Enqueuing task {} with priority {:?}",
            task.id, task.priority
        );

        let mut queue = self.task_queue.write().await;
        queue
            .push(task)
            .map_err(|e| SwarmError::Concurrency(format!("Failed to enqueue task: {}", e)))
    }

    /// Get current edge counts for viscosity calculation
    pub async fn get_edge_counts(&self) -> HashMap<RelationalEdge, i32> {
        let graph_read = self.graph.read().await;
        let mut counts = HashMap::new();

        for edge in &graph_read.edges {
            *counts.entry(edge.edge.clone()).or_insert(0) += 1;
        }

        counts
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> (usize, u64) {
        let queue = self.task_queue.read().await;
        (queue.len(), queue.dropped_tasks())
    }
}

/// Calculate semantic distance between two texts (placeholder)
fn calculate_semantic_distance(text_a: &str, text_b: &str) -> f64 {
    // Simplified Levenshtein distance normalized by length
    let distance = levenshtein_distance(text_a, text_b) as f64;
    let max_len = std::cmp::max(text_a.len(), text_b.len()) as f64;

    if max_len == 0.0 {
        0.0
    } else {
        distance / max_len
    }
}

/// Levenshtein distance implementation
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut dp = vec![vec![0; b_len + 1]; a_len + 1];

    for i in 0..=a_len {
        dp[i][0] = i;
    }

    for j in 0..=b_len {
        dp[0][j] = j;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = std::cmp::min(
                dp[i - 1][j] + 1,
                std::cmp::min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost),
            );
        }
    }

    dp[a_len][b_len]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concourse::types::NodeClass;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "test"), 4);
        assert_eq!(levenshtein_distance("test", ""), 4);
        assert_eq!(levenshtein_distance("same", "same"), 0);
    }

    #[test]
    fn test_semantic_distance() {
        let dist = calculate_semantic_distance("hello world", "hello world");
        assert_eq!(dist, 0.0);

        let dist = calculate_semantic_distance("abc", "xyz");
        assert!(dist > 0.0);
    }

    #[tokio::test]
    async fn test_function_worker() {
        let (tx, _rx) = mpsc::channel(10);
        let worker = FunctionGemmaWorker::new(0, tx);

        let node_a = crate::concourse::types::Node::new(
            "N_1".to_string(),
            NodeClass::Axiom,
            "BASE CONCEPT".to_string(),
        );

        let node_b = crate::concourse::types::Node::new(
            "N_2".to_string(),
            NodeClass::Hypothesis,
            "EXTENSION OF BASE".to_string(),
        );

        let edge = worker.determine_relationship(&node_a, &node_b);
        assert_eq!(edge.edge, RelationalEdge::Scaffolds);
    }
}
