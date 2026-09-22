# Physis Cortex — physis-core plan

> Docs only. No code, no file edits outside plans. Ponytail: reuse what exists.

## 0. What already exists (measured)

| Need | Existing | Verdict |
|---|---|---|
| Embedding | `physis-core/src/embed.rs` `VectorEmbed` trait + `RandomProjectionEmbedder` | reuse — the deterministic fallback IS the CPU backend |
| ONNX inference | `physis-core/src/embed_onnx.rs` via `ort` 2.0.0-rc.13 | reuse — `ort` supports CPU/CUDA/DirectML; ROCm EP is a dead end (packets/README.md) |
| Model provider | `physis-core/src/model_provider.rs` `ModelProvider` + `Capability` | reuse |
| Bench | `physis-core/src/bench.rs` `run()` | reuse |
| World model | `transform.rs`, `process.rs`, `observe.rs`, `delta_engine.rs` | reuse, untouched |
| IR | `physis-pro/src/intent_ir.rs` `IntentIR` (9 slots) | reuse — extend, do not replace |

## 1. Tasks

### Task 1 — `ComputeBackend` trait + `BackendKind`
- Create: `physis-core/src/backend.rs`
- `pub enum BackendKind { Cpu, Rocm, Cuda }`
- `pub trait ComputeBackend { kind(), available_devices(), load_model(spec, device), capabilities(), memory_info(), benchmark() }`
- Reuse `ort` for CUDA: `SessionBuilder::with_execution_providers([CUDAExecutionProvider::default()])`.
- No `#[cfg(rocm)]` / `#[cfg(cuda)]` in semantic code.

### Task 2 — device discovery
- Create: `physis-core/src/devices.rs` — `DeviceInfo`, `HardwareDiscovery::scan()`.
- NVIDIA: `nvidia-smi` query (name, memory, compute capability) + `ort` CUDA EP probe.
- AMD: `rocm-smi` / `/sys/class/drm`.
- CPU: always.
- `physis-core hardware` subcommand.

### Task 3 — LFM encoder backend
- Create: `physis-core/src/cortex/encoder.rs` — `NeuralRole` enum, `CortexOutput`, `EncoderHead`.
- One shared encoder invocation feeding intent/NER/relation heads (adapters/probes first, no joint training).
- Reuse `OnnxEmbedder` for inference.

### Task 4 — canonical Physis latent
- Create: `physis-core/src/cortex/latent.rs` — `PhysisLatent`, projection adapters `W_e/W_d/W_c/W_g`.
- Identity/linear/MLP; benchmark rather than assume.

### Task 5 — retrieval hierarchy
- Create: `physis-core/src/cortex/retrieval.rs` — exact → structural → sparse → dense ANN → ColBERT → graph constraints.
- Reuse `bench.rs` + `nav::query`.

### Task 6 — typed resolver + router
- Create: `physis-core/src/cortex/router.rs` — `Intent` enum, `RoutingDecision`, ladder.
- Reuse `model_provider.rs` `Capability` for escalation.

### Task 7 — microbench
- Extend `bench.rs`: CPU/ROCm/CUDA table per model × quantization.

### Task 8 — regression tests
- `physis-core/tests/cortex_contract.rs`

## 2. Gate
`cargo test --features embed-onnx cortex_contract` + `cargo clippy --lib -- -D warnings`
