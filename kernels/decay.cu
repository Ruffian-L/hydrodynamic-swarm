// kernels/decay.cu
// Metabolic Decay Kernel for KV-Cache on GB10 Unified Memory
// Applies per-element decay based on timestamp/importance map
// Compile: nvcc -arch=sm_90 -c kernels/decay.cu -o decay.o --ptx

extern "C" __global__ void metabolic_decay(float* kv_cache, const float* decay_map, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        kv_cache[idx] *= decay_map[idx];  // Exponential fade for old memories
    }
}

extern "C" __global__ void apply_viscosity(float* scores, float phi, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scores[idx] = scores[idx] / phi;  // Variable Viscosity Φ
    }
}