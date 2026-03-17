use nalgebra::{Matrix2, Vector2};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct CognitiveState {
    pub omega: f64,
    pub k_coupling: f64,
    pub phase_phi: f64,
    pub gamma: f64,
    pub b_z: f64,
}

impl CognitiveState {
    pub fn new() -> Self {
        Self {
            omega: 0.8,
            k_coupling: 1.0,
            phase_phi: 0.0,
            gamma: 1.2,
            b_z: 0.5,
        }
    }

    pub fn update_from_edges(
        &mut self,
        contradiction_count: i32,
        synthesis_count: i32,
        _total_nodes: usize,
    ) {
        self.omega = 0.8 + (contradiction_count as f64) * 0.1;
        self.k_coupling = 1.0 + (synthesis_count as f64) * 0.2;
        self.gamma = 1.2 + (synthesis_count as f64) * 0.5;
        self.b_z = 0.5 + (contradiction_count as f64) * 0.3;
    }

    /// I. Phase Equation (Kuramoto–Adler Reduction)
    pub fn check_phase_lock(&self) -> Result<f64, f64> {
        if self.omega.powi(2) <= self.k_coupling.powi(2) {
            let phi_star = (self.omega / self.k_coupling).asin();
            Ok(phi_star)
        } else {
            let period = (2.0 * PI) / (self.omega.powi(2) - self.k_coupling.powi(2)).sqrt();
            Err(period)
        }
    }

    /// II. Memory Solidification (Ginzburg–Landau Sector)
    pub fn kink_solution(current_r: f64, critical_r: f64) -> f64 {
        let xi = 2.0_f64.sqrt();
        ((current_r - critical_r) / xi).tanh()
    }

    /// III & IV. Coupled Steady-State (Damped Cyclotron Motion)
    pub fn calculate_kinematics(
        &self,
        f_w: Vector2<f64>,
        f_m: Vector2<f64>,
        alpha: f64,
        phi_star: f64,
    ) -> (Vector2<f64>, f64) {
        let prefactor = 1.0 / (self.gamma.powi(2) + self.b_z.powi(2));
        let b_matrix = Matrix2::new(self.gamma, -self.b_z, self.b_z, self.gamma);
        let phase_vector = Vector2::new(phi_star.cos(), phi_star.sin());
        let forces = f_w + f_m + (phase_vector * alpha);
        let v_inf = prefactor * b_matrix * forces;
        let tan_delta = (self.b_z / self.gamma).atan();
        (v_inf, tan_delta)
    }

    /// V. Stability (Lyapunov Analysis)
    pub fn is_lyapunov_stable(&self) -> bool {
        self.gamma > 0.0
    }
}

#[derive(Debug, Clone)]
pub struct VolumetricGovernor {
    pub k_constant: f64,
    pub phi_threshold: f64,
    pub history_window: usize,
    pub critical_radius: f64,
}

impl Default for VolumetricGovernor {
    fn default() -> Self {
        Self {
            k_constant: 0.5,
            phi_threshold: 5.0,
            history_window: 3,
            critical_radius: 5.0,
        }
    }
}

impl VolumetricGovernor {
    pub fn calculate_viscosity(
        &self,
        edge_counts: &[(super::types::RelationalEdge, i32)],
        node_count: usize,
        delta_c: f64,
    ) -> f64 {
        let net_heat: i32 = edge_counts
            .iter()
            .map(|(edge, count)| edge.weight() * count)
            .sum();
        let v_nodes = std::cmp::max(1, node_count) as f64;
        (net_heat as f64 / v_nodes) * std::f64::consts::E.powf(self.k_constant * delta_c)
    }
}
