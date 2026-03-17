//! 9-Node Matrix: Orchestration layer for the complete Swarm
//!
//! Coordinates:
//! - 3 Embed Gemmas (Intake Manifold)
//! - 5 Function Gemmas (Weaver Pool)  
//! - 1 Prime Governor (Splat Governor)
//!
//! Implements the hydrodynamic flow of Flux through the system with
//! natural backpressure via MPSC channels.

use super::embed::{EmbedAlpha, EmbedBeta, EmbedGamma, EmbedManager};
use super::function::{ActiveGraph, FunctionManager};
use super::governor::{ActiveCell, PrimeGovernor};
use super::physics::{CognitiveState, VolumetricGovernor};
use super::types::{FluxTuple, Node};
use super::{SwarmError, SwarmResult};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use tracing::{error, info};

/// Configuration for the 9-Node Matrix
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    pub embed_workers: usize,
    pub function_workers: usize,
    pub channel_buffer: usize,
    pub governor_poll_ms: u64,
    pub phi_threshold: f64,
    pub k_constant: f64,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            embed_workers: 3,
            function_workers: 5,
            channel_buffer: 100,
            governor_poll_ms: 100,
            phi_threshold: 5.0,
            k_constant: 0.5,
        }
    }
}

/// Complete 9-Node Swarm Matrix
#[allow(dead_code)]
pub struct SwarmMatrix {
    config: SwarmConfig,
    embed_manager: EmbedManager,
    function_manager: FunctionManager,
    active_cell: Arc<RwLock<ActiveCell>>,
    cognitive_state: Arc<RwLock<CognitiveState>>,
    node_tx: mpsc::Sender<Node>,
    edge_tx: mpsc::Sender<FluxTuple>,
    shutdown_tx: mpsc::Sender<()>,
}

impl SwarmMatrix {
    /// Initialize the complete 9-Node Matrix
    pub async fn new(config: SwarmConfig) -> SwarmResult<Self> {
        info!("Initializing 9-Node Swarm Matrix...");

        // Create channels for inter-component communication
        let (node_tx, node_rx) = mpsc::channel(config.channel_buffer);
        let (edge_tx, edge_rx) = mpsc::channel(config.channel_buffer);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        let (taco_tx, taco_rx) = mpsc::channel(config.channel_buffer);

        // Initialize shared state
        let active_cell = Arc::new(RwLock::new(ActiveCell::new()));
        let cognitive_state = Arc::new(RwLock::new(CognitiveState::new()));
        let active_graph = Arc::new(RwLock::new(ActiveGraph::new()));

        // Initialize Embed Gemmas (3 instances)
        info!("Initializing Embed Gemmas...");

        let embed_alpha = Arc::new(EmbedAlpha::new()?);
        let embed_beta = Arc::new(EmbedBeta::new()?);
        let embed_gamma = Arc::new(EmbedGamma::new()?);

        let embed_manager =
            EmbedManager::new(embed_alpha, embed_beta, embed_gamma, node_tx.clone());

        // Initialize Function Gemmas (5-instance worker pool)
        info!("Initializing Function Gemmas...");
        let function_manager =
            FunctionManager::new(active_graph, cognitive_state.clone(), edge_tx.clone());

        // Initialize Prime Governor
        info!("Initializing Prime Governor...");
        let governor = PrimeGovernor::new(active_cell.clone(), cognitive_state.clone())
            .with_taco_writer(taco_tx);

        // Spawn background tasks
        let _governor_handle = tokio::spawn(async move {
            if let Err(e) = governor.run(edge_rx, shutdown_rx).await {
                error!("Governor error: {}", e);
            }
        });

        // Spawn node processor
        let function_manager_clone = function_manager.clone();
        let _node_processor_handle = tokio::spawn(async move {
            process_nodes(node_rx, function_manager_clone).await;
        });

        // Spawn TACO.md writer (simplified)
        let _taco_writer_handle = tokio::spawn(async move {
            process_taco_entries(taco_rx).await;
        });

        Ok(Self {
            config,
            embed_manager,
            function_manager,
            active_cell,
            cognitive_state,
            node_tx,
            edge_tx,
            shutdown_tx,
        })
    }

    /// Ingest raw Flux into the Swarm
    pub async fn ingest_flux(&self, raw_text: &str) -> SwarmResult<()> {
        self.embed_manager.process_flux(raw_text).await
    }

    /// Get current Swarm metrics
    pub async fn get_metrics(&self) -> SwarmMetrics {
        let cell = self.active_cell.read().await;
        let state = self.cognitive_state.read().await;

        let edge_counts = cell.get_edge_counts_vec();
        let viscosity = VolumetricGovernor::default().calculate_viscosity(
            &edge_counts,
            cell.node_count(),
            cell.calculate_delta_c(),
        );

        SwarmMetrics {
            node_count: cell.node_count(),
            edge_count: cell.edges.len(),
            viscosity,
            omega: state.omega,
            k_coupling: state.k_coupling,
            gamma: state.gamma,
            b_z: state.b_z,
        }
    }

    /// Gracefully shutdown the Swarm
    pub async fn shutdown(self) -> SwarmResult<()> {
        info!("Initiating graceful Swarm shutdown...");

        // Send shutdown signal to Governor
        self.shutdown_tx.send(()).await.map_err(|e| {
            SwarmError::Concurrency(format!("Failed to send shutdown signal: {}", e))
        })?;

        // Wait a moment for components to clean up
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        info!("Swarm shutdown complete");
        Ok(())
    }
}

/// Process nodes from Embed Gemmas through Function Gemmas
async fn process_nodes(mut node_rx: mpsc::Receiver<Node>, function_manager: FunctionManager) {
    while let Some(node) = node_rx.recv().await {
        if let Err(e) = function_manager.process_node(node).await {
            error!("Error processing node: {}", e);
        }
    }
}

/// Process TACO.md entries (simplified file writer)
async fn process_taco_entries(mut taco_rx: mpsc::Receiver<String>) {
    // In production, this would write to actual TACO.md file
    // For now, just log the entries
    while let Some(entry) = taco_rx.recv().await {
        info!("TACO.md entry: {}", entry);
    }
}

/// Swarm performance metrics
#[derive(Debug, Clone)]
pub struct SwarmMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub viscosity: f64,
    pub omega: f64,
    pub k_coupling: f64,
    pub gamma: f64,
    pub b_z: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swarm_initialization() {
        let config = SwarmConfig::default();
        let result = SwarmMatrix::new(config).await;

        // Should fail because candle Gemma models aren't actually loaded
        // but we can verify the structure compiles
        assert!(result.is_err() || result.is_ok());
    }

    #[cfg(feature = "with-candle")]
    #[tokio::test]
    async fn test_swarm_ingest_flux_with_real_models() {
        let config = SwarmConfig::default();
        let swarm = SwarmMatrix::new(config)
            .await
            .expect("Swarm initialization should succeed with real models");
        // Ingest a small piece of flux
        let result = swarm.ingest_flux("test code").await;
        assert!(result.is_ok());
    }
}
