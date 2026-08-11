# Vanguard

> **An Auto-Optimizing GPT-2 124M Inference Engine & ML Systems Engineering Curriculum**

Vanguard is an educational, high-performance C++/CUDA inference runtime for **GPT-2 124M**. It serves as a hands-on vehicle for mastering **ML/AI Systems Engineering**—from raw tensor memory layouts and progressive CUDA kernel optimizations to roofline profiling, KV caching, and hardware-aware execution scheduling.

---

## Technical Architecture

```text
                                  Vanguard Runtime
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  Model & Weights                 Runtime Scheduler               Profiling System
(mmap binary loader)             (Hardware Detector)            (CUDA Events, Metrics)
  [GPT-2 124M FP32]                      │                       [Bandwidth, TFLOPs]
                                         ▼
                                  Kernel Selector
                             (Auto-tuner & Heuristics)
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
                  CPU Reference                     CUDA Engine
                (Golden Baseline)                   (GPU Runtime)
                        │                                 │
           ┌────────────┴────────────┐       ┌────────────┴────────────┐
           ▼                         ▼       ▼                         ▼
      CPU Kernels                KV Cache   CUDA Kernels          Workspace Allocator
  (Eigen/OpenMP/Plain C++)     (Contiguous) (Modular Kernels)    (Zero-allocation loop)
                                             ├── MatMul (Naive -> Tiled -> Tensor Core)
                                             ├── LayerNorm (Naive -> Warp-reduced)
                                             ├── GELU & Softmax (Fused)
                                             └── Causal Attention (KV-Cached)
```

---

## Directory Structure

```text
Vanguard/
├── docs/                              # Architecture specs & engineering principles
├── research/                          # Systems research & notes
├── assets/                            # Project media & documentation assets
├── prototypes/                        # Exploratory prototypes & scripts
│   ├── engine/                        # Early CUDA single-token engine prototype
│   ├── gpt2_export/                   # PyTorch model weight export prototype
│   └── hardware_manager/              # Standalone GPU telemetry scanner prototype
│
├── src/                               # Vanguard Core & Backend Source Code
│   ├── core/                          # Platform-independent core components
│   │   ├── model/                     # Model weights loader & BPE tokenizer
│   │   ├── inference/                 # Inference orchestration (future)
│   │   ├── memory/                    # Memory managers & workspace allocators (future)
│   │   ├── scheduler/                 # Kernel execution scheduling (future)
│   │   ├── optimizer/                 # Graph & kernel optimizers (future)
│   │   ├── profiling/                 # Cross-platform profiling primitives (future)
│   │   ├── tensor.hpp                 # Multi-dimensional TensorView templates
│   │   └── types.hpp                  # GPT-2 124M specs & system precision types
│   │
│   ├── backends/                      # Platform- & hardware-specific backends
│   │   ├── cpu/                       # Golden Reference CPU Engine & Kernels
│   │   ├── cuda/                      # High-Performance CUDA Engine & KV-Cache
│   │   └── metal/                     # Apple Metal GPU Backend (future)
│   │
│   └── runtime/                       # Hardware probe & system telemetry
│
├── apps/                              # User-facing applications
│   ├── cli/                           # Desktop CLI inference engine
│   └── ios/                           # Future Swift/SwiftUI iOS application
│
├── tests/                             # Test harnesses
│   ├── unit/                          # Component unit tests (future)
│   ├── integration/                   # CPU/CUDA golden assertion correctness tests
│   └── benchmarks/                    # Performance & latency benchmarks (future)
│
├── scripts/                           # PyTorch export & verification python scripts
├── CMakeLists.txt                     # Modern CMake build configuration
├── Makefile                           # Unified Makefile build system
├── README.md                          # Project overview & roadmap
└── .gitignore
```

---

## Quickstart & Build Instructions

### Prerequisites
- Linux (x86_64)
- `g++` (C++17 support)
- NVIDIA CUDA Toolkit (`nvcc`, CUDA 11.0+)
- Python 3 + PyTorch & Transformers (`pip install torch transformers numpy`)

### 1. Build and Run CPU Reference Engine Correctness Test
```bash
make cpu_test
```
*Outputs verification against PyTorch golden logits ($\le 10^{-3}$ tolerance).*

### 2. Build and Run CUDA Engine Test
```bash
make cuda_test
```

### 3. Generate PyTorch Golden Tensors
```bash
python3 scripts/verify_correctness.py --prompt "The square root of 4"
```

---

## Systems Learning Roadmap

1. **Milestone 1:** Core Baseline Restructure & Numerical Correctness Verification Harness *(Completed)*
2. **Milestone 2:** Persistent KV Cache Manager & Prefill vs Decode Phase Separation
3. **Milestone 3:** Zero-Allocation Workspace Memory Allocator & Host/Device Memory Profiling
4. **Milestone 4:** Progressive CUDA MatMul Optimization (Naive $\rightarrow$ Shared Memory Tiling $\rightarrow$ Vectorized $\rightarrow$ cuBLAS)
5. **Milestone 5:** Fused Kernels & Reduction Primitives (LayerNorm, GELU, Softmax)
6. **Milestone 6:** Auto-Optimizing Runtime & Hardware-Aware Kernel Selector
7. **Milestone 7:** Numerical Precision & INT8 Quantization Exploration

---

## License
MIT License.
