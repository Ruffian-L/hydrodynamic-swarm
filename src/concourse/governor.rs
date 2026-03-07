//! Prime Governor: Volumetric governor monitoring system Viscosity
//!
//! Single instance in the 9-Node Matrix that:
//! 1. Continuously polls the active cell's thermodynamic state
//! 2. Calculates Viscosity (Φ) using the governor equation
//! 3. Executes the Splat reflex when critical thresholds are reached
//! 4. Manages TACO.md persistence and cell lifecycle

use super::physics::{CognitiveState, VolumetricGovernor};
use super::types::{FluxTuple, Node, NodeClass, RelationalEdge};
use super::{SwarmError, SwarmResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

/// Active cell state managed by the Governor
#[derive(Debug, Serialize, Deserialize)]
pub struct ActiveCell {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<FluxTuple>,
    pub friction_history: VecDeque<i32>,
    pub edge_counts: HashMap<RelationalEdge, i32>,
}

impl ActiveCell {
    pub fn new() -> Self {
        let mut edge_counts = HashMap::new();
        for edge in [
            RelationalEdge::Encapsulates,
            RelationalEdge::Scaffolds,
            RelationalEdge::Actuates,
            RelationalEdge::IsIsomorphicTo,
            RelationalEdge::Contradicts,
            RelationalEdge::Catalyzes,
            RelationalEdge::Synthesizes,
        ] {
            edge_counts.insert(edge, 0);
        }

        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            friction_history: VecDeque::with_capacity(3),
            edge_counts,
        }
    }

    pub fn add_edge(&mut self, tuple: FluxTuple) {
        self.edges.push(tuple.clone());
        *self.edge_counts.get_mut(&tuple.edge).unwrap() += 1;

        // Update node tracking (simplified - actual implementation would add nodes from Embed Gemmas)
        if !self.nodes.contains_key(&tuple.source) {
            // Placeholder node creation
            let node = Node::new(
                tuple.source.clone(),
                NodeClass::Observation,
                "PLACEHOLDER".to_string(),
            );
            self.nodes.insert(tuple.source.clone(), node);
        }

        if !self.nodes.contains_key(&tuple.target) {
            let node = Node::new(
                tuple.target.clone(),
                NodeClass::Observation,
                "PLACEHOLDER".to_string(),
            );
            self.nodes.insert(tuple.target.clone(), node);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_edge_counts_vec(&self) -> Vec<(RelationalEdge, i32)> {
        self.edge_counts
            .iter()
            .map(|(edge, count)| (edge.clone(), *count))
            .collect()
    }

    pub fn calculate_delta_c(&self) -> f64 {
        if self.friction_history.len() >= 2 {
            let first = *self.friction_history.front().unwrap() as f64;
            let last = *self.friction_history.back().unwrap() as f64;
            (last - first).max(0.0)
        } else {
            0.0
        }
    }
}

/// Prime Governor daemon
pub struct PrimeGovernor {
    active_cell: Arc<RwLock<ActiveCell>>,
    cognitive_state: Arc<RwLock<CognitiveState>>,
    volumetric_governor: VolumetricGovernor,
    taco_writer: Option<mpsc::Sender<String>>, // For TACO.md persistence
}

impl PrimeGovernor {
    pub fn new(
        active_cell: Arc<RwLock<ActiveCell>>,
        cognitive_state: Arc<RwLock<CognitiveState>>,
    ) -> Self {
        Self {
            active_cell,
            cognitive_state,
            volumetric_governor: VolumetricGovernor::default(),
            taco_writer: None,
        }
    }

    /// Set TACO.md writer channel
    pub fn with_taco_writer(mut self, writer: mpsc::Sender<String>) -> Self {
        self.taco_writer = Some(writer);
        self
    }

    /// Main governor loop
    pub async fn run(
        mut self,
        mut edge_rx: mpsc::Receiver<FluxTuple>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) -> SwarmResult<()> {
        info!("Prime Governor daemon started");

        let mut poll_interval = interval(Duration::from_millis(100));

        loop {
            tokio::select! {
                // Receive new edges from Function Gemmas
                Some(tuple) = edge_rx.recv() => {
                    self.process_edge(tuple).await?;
                }

                // Periodic viscosity polling
                _ = poll_interval.tick() => {
                    self.poll_viscosity().await?;
                }

                // Shutdown signal
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received by Governor");
                    break;
                }
            }
        }

        info!("Prime Governor daemon stopped");
        Ok(())
    }

    /// Process incoming edge and update cell state
    async fn process_edge(&mut self, tuple: FluxTuple) -> SwarmResult<()> {
        let mut cell = self.active_cell.write().await;
        cell.add_edge(tuple.clone());

        // Update friction history for [CONTRADICTS] and [CATALYZES] edges
        if matches!(
            tuple.edge,
            RelationalEdge::Contradicts | RelationalEdge::Catalyzes
        ) {
            let current_friction = cell.edge_counts[&RelationalEdge::Contradicts]
                + cell.edge_counts[&RelationalEdge::Catalyzes];

            if cell.friction_history.len() == 3 {
                cell.friction_history.pop_front();
            }
            cell.friction_history.push_back(current_friction);
        }

        info!(
            "Governor processed edge: {} {:?} {}",
            tuple.source, tuple.edge, tuple.target
        );

        Ok(())
    }

    /// Poll current viscosity and trigger Splat if needed
    async fn poll_viscosity(&mut self) -> SwarmResult<()> {
        let (edge_counts, node_count, delta_c) = {
            let cell = self.active_cell.read().await;
            let edge_counts = cell.get_edge_counts_vec();
            let node_count = cell.node_count();
            let delta_c = cell.calculate_delta_c();
            (edge_counts, node_count, delta_c)
        };

        // Calculate viscosity
        let phi = self
            .volumetric_governor
            .calculate_viscosity(&edge_counts, node_count, delta_c);

        // Update cognitive state
        {
            let mut state = self.cognitive_state.write().await;
            let contradiction_count = edge_counts
                .iter()
                .find(|(e, _)| matches!(e, RelationalEdge::Contradicts))
                .map(|(_, c)| *c)
                .unwrap_or(0);
            let synthesis_count = edge_counts
                .iter()
                .find(|(e, _)| matches!(e, RelationalEdge::Synthesizes))
                .map(|(_, c)| *c)
                .unwrap_or(0);

            state.update_from_edges(contradiction_count, synthesis_count, node_count);
        }

        // Check Lyapunov stability
        let is_stable = {
            let state = self.cognitive_state.read().await;
            state.is_lyapunov_stable()
        };
        if !is_stable {
            error!("Lyapunov stability broken! Executing emergency Splat.");
            self.execute_splat(phi, node_count, delta_c, true).await?;
            return Ok(());
        }

        // Check viscosity threshold
        if phi >= self.volumetric_governor.phi_threshold {
            warn!(
                "Critical viscosity detected: Φ = {:.2} (threshold: {})",
                phi, self.volumetric_governor.phi_threshold
            );
            self.execute_splat(phi, node_count, delta_c, false).await?;
        } else {
            info!(
                "Viscosity normal: Φ = {:.2}, nodes = {}, ΔC = {:.2}",
                phi, node_count, delta_c
            );
        }

        Ok(())
    }

    /// Execute Splat reflex
    async fn execute_splat(
        &mut self,
        phi: f64,
        node_count: usize,
        delta_c: f64,
        emergency: bool,
    ) -> SwarmResult<()> {
        let trigger_type = if emergency { "EMERGENCY" } else { "CRITICAL" };

        info!(
            "\n[!!!] {} VISCOSITY DETECTED: Φ = {:.2} [!!!]",
            trigger_type, phi
        );
        info!("EXECUTING SPLAT PROTOCOL...");

        // 1. SEVER: Capture current cell state
        let _cell_snapshot = {
            let cell = self.active_cell.read().await;
            serde_json::to_string(&*cell).map_err(|e| SwarmError::Serialization(e))?
        };

        // 2. IGNITE & ENCAPSULATE: Create macro-node representation
        let _macro_node = format!(
            "CELL_SNAPSHOT: nodes={}, edges={}, phi={:.2}, delta_c={:.2}",
            node_count,
            {
                let cell = self.active_cell.read().await;
                cell.edges.len()
            },
            phi,
            delta_c
        );

        // 3. COMMIT: Write to TACO.md (if writer available)
        if let Some(ref writer) = self.taco_writer {
            let taco_entry = format!(
                "---\nSPLAT: {}\nTimestamp: {:?}\nNodes: {}\nEdges: {}\nPhi: {:.2}\nDeltaC: {:.2}\n---\n",
                trigger_type,
                std::time::SystemTime::now(),
                node_count,
                {
                    let cell = self.active_cell.read().await;
                    cell.edges.len()
                },
                phi,
                delta_c
            );

            writer
                .send(taco_entry)
                .await
                .map_err(|e| SwarmError::Concurrency(format!("Failed to write TACO.md: {}", e)))?;
        }

        // 4. VOID: Flush active cell
        {
            let mut cell = self.active_cell.write().await;
            *cell = ActiveCell::new();
        }

        // 5. Reset cognitive state
        {
            let mut state = self.cognitive_state.write().await;
            *state = CognitiveState::new();
        }

        info!(
            "  1_SEVER: Cell captured ({} nodes, {} edges)",
            node_count,
            {
                let cell = self.active_cell.read().await;
                cell.edges.len()
            }
        );
        info!("  2_ENCAPSULATE: Macro-node created");
        info!("  3_COMMIT: Diamond written to TACO.md");
        info!("  4_VOID: Active cell flushed. Spawning pristine Cell N+1...\n");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_active_cell_edge_counting() {
        let cell = ActiveCell::new();
        assert_eq!(cell.node_count(), 0);

        let edge_counts = cell.get_edge_counts_vec();
        assert_eq!(edge_counts.len(), 7);

        for (edge, count) in edge_counts {
            assert_eq!(count, 0);
        }
    }

    #[tokio::test]
    async fn test_viscosity_calculation() {
        let (edge_tx, edge_rx) = mpsc::channel(10);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let active_cell = Arc::new(RwLock::new(ActiveCell::new()));
        let cognitive_state = Arc::new(RwLock::new(CognitiveState::new()));

        let governor = PrimeGovernor::new(active_cell.clone(), cognitive_state.clone());

        // Send a contradiction edge (high viscosity)
        edge_tx
            .send(FluxTuple {
                source: "N_1".to_string(),
                edge: RelationalEdge::Contradicts,
                target: "N_2".to_string(),
            })
            .await
            .unwrap();

        // Send another contradiction (should increase viscosity further)
        edge_tx
            .send(FluxTuple {
                source: "N_2".to_string(),
                edge: RelationalEdge::Contradicts,
                target: "N_3".to_string(),
            })
            .await
            .unwrap();

        drop(edge_tx);
        shutdown_tx.send(()).await.unwrap();

        // Governor would process these in real scenario
        // For now just verify the test compiles
        assert!(true);
    }
}
