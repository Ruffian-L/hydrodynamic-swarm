#![allow(dead_code)]
//! Physics Backend Abstraction
//!
//! Defines the `PhysicsBackend` trait for GPU-accelerated physics computations.
//! Provides a `CpuBackend` that wraps the existing field/memory code verbatim,
//! and a `MetalBackend` (behind `metal-compute` feature) with wgpu WGSL compute shaders.
//!
//! All call sites (niodoo, dream, ridge) use `&dyn PhysicsBackend` so the GPU
//! path is a drop-in replacement with zero API changes.

use candle_core::{Result, Tensor};

use crate::field::ContinuousField;
use crate::memory::SplatMemory;

// ---------------------------------------------------------------
// Trait
// ---------------------------------------------------------------

/// Abstraction over CPU vs GPU physics computations.
///
/// Each method corresponds to a hot-path operation in the steering loop.
/// Implementations must produce identical results (within floating-point tolerance).
pub trait PhysicsBackend {
    /// Gradient of the continuous field density at `pos` (shape `(D,)`).
    /// Replaces `ContinuousField::probe_gradient`.
    fn field_gradient(&self, field: &ContinuousField, pos: &Tensor) -> Result<Tensor>;

    /// Aggregate Gaussian force from all splats at `pos` (shape `(D,)`).
    /// Replaces `SplatMemory::query_force`.
    fn splat_force(&self, memory: &SplatMemory, pos: &Tensor) -> Result<Tensor>;

    /// Batch field gradient for multiple positions (shape `(M, D)` -> `(M, D)`).
    /// Used by micro-dream forward projection and ridge-running ensembles.
    fn batch_field_gradient(&self, field: &ContinuousField, positions: &Tensor) -> Result<Tensor>;

    /// Name string for telemetry logging.
    fn name(&self) -> &'static str;

    /// Top-K approximate gradient: only consider the K nearest field points.
    /// Default implementation falls back to exact gradient.
    fn field_gradient_topk(
        &self,
        field: &ContinuousField,
        pos: &Tensor,
        k: usize,
    ) -> Result<Tensor> {
        let _ = k; // unused in default -- exact gradient
        self.field_gradient(field, pos)
    }
}

// ---------------------------------------------------------------
// CPU Backend (wraps existing code verbatim)
// ---------------------------------------------------------------

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicsBackend for CpuBackend {
    fn field_gradient(&self, field: &ContinuousField, pos: &Tensor) -> Result<Tensor> {
        field.probe_gradient(pos)
    }

    fn splat_force(&self, memory: &SplatMemory, pos: &Tensor) -> Result<Tensor> {
        memory.query_force(pos)
    }

    fn batch_field_gradient(&self, field: &ContinuousField, positions: &Tensor) -> Result<Tensor> {
        let m = positions.dim(0)?;
        if m == 0 {
            let d = positions.dim(1)?;
            return Tensor::zeros(&[0, d], candle_core::DType::F32, &field.device);
        }
        let mut grads = Vec::with_capacity(m);
        for i in 0..m {
            let pos_i = positions.get(i)?;
            let grad_i = field.probe_gradient(&pos_i)?.unsqueeze(0)?;
            grads.push(grad_i);
        }
        Tensor::cat(&grads, 0)
    }

    fn name(&self) -> &'static str {
        "CPU"
    }

    fn field_gradient_topk(
        &self,
        field: &ContinuousField,
        pos: &Tensor,
        k: usize,
    ) -> Result<Tensor> {
        field.probe_gradient_topk(pos, k)
    }
}

// ---------------------------------------------------------------
// Metal Backend (behind feature flag)
// ---------------------------------------------------------------

#[cfg(feature = "metal-compute")]
mod metal_backend {
    use super::*;

    /// WGSL compute shader: field gradient.
    ///
    /// Each workgroup computes the gradient contribution from all field points
    /// for one query position. Uses padded vec4 layout for Metal alignment.
    ///
    /// The shader processes D dimensions in chunks of 4 (vec4<f32>).
    /// For D=4096, that's 1024 vec4 elements per position.
    const FIELD_GRADIENT_SHADER: &str = r#"
// Uniforms: dimensions and kernel parameters
// Padded to 32 bytes (8 x u32) for 16-byte WGSL uniform alignment.
struct Params {
    n_points: u32,      // number of field points
    n_queries: u32,     // number of query positions
    dim_vec4: u32,      // D / 4 (number of vec4 elements per position)
    sigma_sq: f32,      // kernel_sigma^2
    scale: f32,         // 2.0 / sigma_sq
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Field positions: flat array of vec4, layout [n_points * dim_vec4]
@group(0) @binding(1) var<storage, read> field_positions: array<vec4<f32>>;
// Query positions: flat array of vec4, layout [n_queries * dim_vec4]
@group(0) @binding(2) var<storage, read> query_positions: array<vec4<f32>>;
// Output gradients: flat array of vec4, layout [n_queries * dim_vec4]
@group(0) @binding(3) var<storage, read_write> gradients: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    if query_idx >= params.n_queries {
        return;
    }

    let q_offset = query_idx * params.dim_vec4;

    // For each field point, compute Gaussian kernel weight and accumulate gradient
    for (var p = 0u; p < params.n_points; p++) {
        let p_offset = p * params.dim_vec4;

        // Compute ||field_pos - query_pos||^2
        var dist_sq = 0.0;
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = field_positions[p_offset + d] - query_positions[q_offset + d];
            dist_sq += dot(diff, diff);
        }

        // Gaussian kernel: exp(-dist_sq / sigma_sq)
        let kernel = exp(-dist_sq / params.sigma_sq);

        // Skip if kernel underflows
        if kernel < 1e-30 {
            continue;
        }

        // Accumulate weighted gradient: kernel * (field_pos - query_pos) * scale
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = field_positions[p_offset + d] - query_positions[q_offset + d];
            gradients[q_offset + d] += diff * kernel * params.scale;
        }
    }
}
"#;

    /// WGSL compute shader: splat force aggregation.
    ///
    /// Each invocation processes one query position against all splats.
    /// Splats are stored as: mu (vec4 * dim_vec4), sigma (f32), alpha (f32).
    const SPLAT_FORCE_SHADER: &str = r#"
struct Params {
    n_splats: u32,
    n_queries: u32,
    dim_vec4: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Splat mu positions: flat [n_splats * dim_vec4] vec4s
@group(0) @binding(1) var<storage, read> splat_mu: array<vec4<f32>>;
// Splat sigma values: [n_splats]
@group(0) @binding(2) var<storage, read> splat_sigma: array<f32>;
// Splat alpha values: [n_splats]
@group(0) @binding(3) var<storage, read> splat_alpha: array<f32>;
// Query positions: [n_queries * dim_vec4] vec4s
@group(0) @binding(4) var<storage, read> query_positions: array<vec4<f32>>;
// Output forces: [n_queries * dim_vec4] vec4s
@group(0) @binding(5) var<storage, read_write> forces: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    if query_idx >= params.n_queries {
        return;
    }

    let q_offset = query_idx * params.dim_vec4;

    for (var s = 0u; s < params.n_splats; s++) {
        let s_offset = s * params.dim_vec4;
        let sigma = splat_sigma[s];
        let alpha = splat_alpha[s];
        let sigma_sq = sigma * sigma;

        // Compute ||mu - pos||^2
        var dist_sq = 0.0;
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = splat_mu[s_offset + d] - query_positions[q_offset + d];
            dist_sq += dot(diff, diff);
        }

        // kernel = exp(-dist_sq / sigma_sq)
        let kernel = exp(-dist_sq / sigma_sq);
        let scale = alpha * kernel;

        // force += scale * (mu - pos)
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = splat_mu[s_offset + d] - query_positions[q_offset + d];
            forces[q_offset + d] += diff * scale;
        }
    }
}
"#;

    pub struct MetalBackend {
        device: wgpu::Device,
        queue: wgpu::Queue,
        field_gradient_pipeline: wgpu::ComputePipeline,
        splat_force_pipeline: wgpu::ComputePipeline,
    }

    impl MetalBackend {
        pub fn try_new() -> Option<Self> {
            // Block on async wgpu init -- fine at startup
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                ..Default::default()
            });

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hydrodynamic-swarm-metal"),
                    ..Default::default()
                },
                None,
            ))
            .ok()?;

            // Compile shaders
            let field_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("field_gradient_shader"),
                source: wgpu::ShaderSource::Wgsl(FIELD_GRADIENT_SHADER.into()),
            });

            let splat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("splat_force_shader"),
                source: wgpu::ShaderSource::Wgsl(SPLAT_FORCE_SHADER.into()),
            });

            let field_gradient_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("field_gradient_pipeline"),
                    layout: None, // auto layout
                    module: &field_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let splat_force_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("splat_force_pipeline"),
                    layout: None,
                    module: &splat_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            println!("    [METAL] Adapter: {}", adapter.get_info().name);

            Some(Self {
                device,
                queue,
                field_gradient_pipeline,
                splat_force_pipeline,
            })
        }

        /// Run the field gradient compute shader for `n_queries` query positions.
        fn dispatch_field_gradient(
            &self,
            field_data: &[f32], // flat [N * D] field positions
            query_data: &[f32], // flat [M * D] query positions
            n_points: u32,
            n_queries: u32,
            dim: u32,
            sigma_sq: f32,
        ) -> Vec<f32> {
            use wgpu::util::DeviceExt;

            let dim_vec4 = dim / 4;
            let scale = 2.0f32 / sigma_sq;

            // Uniform params buffer (padded to 32 bytes to match WGSL struct)
            let params = [
                n_points,
                n_queries,
                dim_vec4,
                sigma_sq.to_bits(),
                scale.to_bits(),
                0u32, // _pad1
                0u32, // _pad2
                0u32, // _pad3
            ];
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // Field positions buffer
            let field_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("field_positions"),
                    contents: bytemuck::cast_slice(field_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            // Query positions buffer
            let query_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("query_positions"),
                    contents: bytemuck::cast_slice(query_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            // Output gradient buffer (must be zeroed -- shader uses +=)
            let output_size = (n_queries * dim) as u64 * 4;
            let zeros = vec![0u8; output_size as usize];
            let output_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gradients"),
                    contents: &zeros,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

            // Staging buffer for readback
            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Bind group
            let bind_group_layout = self.field_gradient_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("field_gradient_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: field_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: query_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

            // Encode and submit
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("field_gradient_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("field_gradient_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.field_gradient_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n_queries.div_ceil(256), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            // Read back results
            let slice = staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                if let Err(e) = res {
                    eprintln!("[METAL] field_gradient staging buffer map failed: {e}");
                }
            });
            self.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();

            result
        }

        /// Run the splat force compute shader.
        #[allow(clippy::too_many_arguments)]
        fn dispatch_splat_force(
            &self,
            splat_mu_data: &[f32],    // flat [S * D]
            splat_sigma_data: &[f32], // [S]
            splat_alpha_data: &[f32], // [S]
            query_data: &[f32],       // flat [M * D]
            n_splats: u32,
            n_queries: u32,
            dim: u32,
        ) -> Vec<f32> {
            use wgpu::util::DeviceExt;

            let dim_vec4 = dim / 4;

            // Uniform params
            let params = [n_splats, n_queries, dim_vec4, 0u32]; // _pad = 0
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let mu_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_mu"),
                    contents: bytemuck::cast_slice(splat_mu_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let sigma_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_sigma"),
                    contents: bytemuck::cast_slice(splat_sigma_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let alpha_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_alpha"),
                    contents: bytemuck::cast_slice(splat_alpha_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let query_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("query"),
                    contents: bytemuck::cast_slice(query_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let output_size = (n_queries * dim) as u64 * 4;
            let zeros = vec![0u8; output_size as usize];
            let output_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("forces"),
                    contents: &zeros,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout = self.splat_force_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat_force_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: mu_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sigma_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: alpha_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: query_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("splat_force_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_force_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.splat_force_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n_queries.div_ceil(256), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            let slice = staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                if let Err(e) = res {
                    eprintln!("[METAL] splat_force staging buffer map failed: {e}");
                }
            });
            self.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();

            result
        }

        /// Convert a Candle Tensor to a flat Vec<f32> for GPU upload.
        fn tensor_to_f32(t: &Tensor) -> Result<Vec<f32>> {
            t.flatten_all()?.to_vec1()
        }

        /// Pad a flat f32 slice to be vec4-aligned (D must be divisible by 4).
        /// For D=4096, this is already aligned. This is a safety check.
        fn ensure_vec4_aligned(dim: usize) -> bool {
            dim.is_multiple_of(4)
        }
    }

    impl PhysicsBackend for MetalBackend {
        fn field_gradient(&self, field: &ContinuousField, pos: &Tensor) -> Result<Tensor> {
            let dim = field.dim;

            // Fall back to CPU if dim is not vec4-aligned
            if !Self::ensure_vec4_aligned(dim) {
                return field.probe_gradient(pos);
            }

            let field_data = Self::tensor_to_f32(&field.positions)?;
            let query_data = Self::tensor_to_f32(pos)?;
            let n_points = field.n_points() as u32;

            let result = self.dispatch_field_gradient(
                &field_data,
                &query_data,
                n_points,
                1,
                dim as u32,
                field.kernel_sigma * field.kernel_sigma,
            );

            // Check for all-zero (underflow case)
            let sum: f32 = result.iter().map(|x| x.abs()).sum();
            if sum < 1e-30 {
                return candle_core::Tensor::zeros(&[dim], candle_core::DType::F32, &field.device);
            }

            Tensor::from_vec(result, dim, &field.device)
        }

        fn splat_force(&self, memory: &SplatMemory, pos: &Tensor) -> Result<Tensor> {
            let splats = memory.splats_ref();
            if splats.is_empty() {
                let dims = pos.dims().to_vec();
                return Tensor::zeros(&dims[..], candle_core::DType::F32, pos.device());
            }

            let dim = pos.dims()[0];
            if !Self::ensure_vec4_aligned(dim) {
                return memory.query_force(pos);
            }

            // Extract splat data
            let n_splats = splats.len();
            let mut mu_data = Vec::with_capacity(n_splats * dim);
            let mut sigma_data = Vec::with_capacity(n_splats);
            let mut alpha_data = Vec::with_capacity(n_splats);

            for splat in splats {
                let mu_flat: Vec<f32> = splat.mu.flatten_all()?.to_vec1()?;
                mu_data.extend_from_slice(&mu_flat);
                sigma_data.push(splat.sigma);
                alpha_data.push(splat.alpha);
            }

            let query_data = Self::tensor_to_f32(pos)?;

            let result = self.dispatch_splat_force(
                &mu_data,
                &sigma_data,
                &alpha_data,
                &query_data,
                n_splats as u32,
                1,
                dim as u32,
            );

            Tensor::from_vec(result, dim, pos.device())
        }

        fn batch_field_gradient(
            &self,
            field: &ContinuousField,
            positions: &Tensor,
        ) -> Result<Tensor> {
            let dim = field.dim;

            if !Self::ensure_vec4_aligned(dim) {
                // Fall back to serial CPU
                let m = positions.dim(0)?;
                if m == 0 {
                    return Tensor::zeros(&[0, dim], candle_core::DType::F32, &field.device);
                }
                let mut grads = Vec::with_capacity(m);
                for i in 0..m {
                    let pos_i = positions.get(i)?;
                    let grad_i = field.probe_gradient(&pos_i)?.unsqueeze(0)?;
                    grads.push(grad_i);
                }
                return Tensor::cat(&grads, 0);
            }

            let n_queries = positions.dim(0)? as u32;
            let field_data = Self::tensor_to_f32(&field.positions)?;
            let query_data = Self::tensor_to_f32(positions)?;

            let result = self.dispatch_field_gradient(
                &field_data,
                &query_data,
                field.n_points() as u32,
                n_queries,
                dim as u32,
                field.kernel_sigma * field.kernel_sigma,
            );

            Tensor::from_vec(result, (n_queries as usize, dim), &field.device)
        }

        fn name(&self) -> &'static str {
            "Metal Compute"
        }
    }
}

// ---------------------------------------------------------------
// Backend selection helper
// ---------------------------------------------------------------

/// Select the best available physics backend at runtime.
///
/// With `metal-compute` feature: tries MetalBackend first, falls back to CPU.
/// Without feature: always returns CpuBackend.
pub fn select_backend() -> Box<dyn PhysicsBackend> {
    #[cfg(feature = "metal-compute")]
    {
        if let Some(metal) = metal_backend::MetalBackend::try_new() {
            println!("[*] Physics backend: Metal Compute (wgpu)");
            return Box::new(metal);
        }
        println!("[*] Metal compute init failed, falling back to CPU physics");
    }

    println!("[*] Physics backend: CPU");
    Box::new(CpuBackend::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn cpu_backend_field_gradient_matches_direct() {
        let device = Device::Cpu;
        let positions = Tensor::randn(0.0f32, 1.0, &[10, 4], &device).unwrap();
        let field = crate::field::ContinuousField {
            device: device.clone(),
            positions,
            kernel_sigma: 1.0,
            dim: 4,
        };

        let backend = CpuBackend::new();
        let pos = Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], &device).unwrap();

        let direct = field.probe_gradient(&pos).unwrap();
        let via_backend = backend.field_gradient(&field, &pos).unwrap();

        let diff: f32 = (&direct - &via_backend)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            diff < 1e-6,
            "CPU backend should match direct, diff={}",
            diff
        );
    }

    #[test]
    fn cpu_backend_splat_force_matches_direct() {
        let device = Device::Cpu;
        let mut memory = crate::memory::SplatMemory::new(device.clone());
        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory.add_splat(crate::splat::Splat::new(mu, 1.0, 3.0));

        let backend = CpuBackend::new();
        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();

        let direct = memory.query_force(&pos).unwrap();
        let via_backend = backend.splat_force(&memory, &pos).unwrap();

        let diff: f32 = (&direct - &via_backend)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            diff < 1e-6,
            "CPU backend splat force should match direct, diff={}",
            diff
        );
    }

    #[test]
    fn cpu_backend_batch_gradient_shape() {
        let device = Device::Cpu;
        let positions = Tensor::randn(0.0f32, 1.0, &[5, 4], &device).unwrap();
        let field = crate::field::ContinuousField {
            device: device.clone(),
            positions,
            kernel_sigma: 1.0,
            dim: 4,
        };

        let backend = CpuBackend::new();
        let queries = Tensor::randn(0.0f32, 1.0, &[3, 4], &device).unwrap();
        let result = backend.batch_field_gradient(&field, &queries).unwrap();
        assert_eq!(result.dims(), &[3, 4]);
    }

    /// GPU parity test -- requires `metal-compute` feature and available GPU.
    /// Runs only when compiled with `cargo test --features metal-compute`.
    #[test]
    #[cfg(feature = "metal-compute")]
    fn metal_cpu_field_gradient_parity() {
        let device = Device::Cpu;
        let dim = 8; // small dim, vec4-aligned
        let positions = Tensor::randn(0.0f32, 1.0, &[20, dim], &device).unwrap();
        let field = crate::field::ContinuousField {
            device: device.clone(),
            positions,
            kernel_sigma: 1.0,
            dim,
        };

        let cpu = CpuBackend::new();
        let metal = match metal_backend::MetalBackend::try_new() {
            Some(m) => m,
            None => {
                eprintln!("    [SKIP] Metal GPU not available for parity test");
                return;
            }
        };

        let pos = Tensor::randn(0.0f32, 1.0, &[dim], &device).unwrap();
        let cpu_result = cpu.field_gradient(&field, &pos).unwrap();
        let metal_result = metal.field_gradient(&field, &pos).unwrap();

        let diff: f32 = (&cpu_result - &metal_result)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();

        // Allow some FP tolerance since GPU uses f32 with different instruction ordering
        assert!(
            diff < 0.1,
            "CPU vs Metal field gradient should match within tolerance, diff={}",
            diff
        );
    }
}
