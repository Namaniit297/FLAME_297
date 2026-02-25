# FLAME_297 FLAME / DynKV prototype — README

Overview:
- This repo provides:
  - Akita-side prototype primitives: fragment directory and PMEngine stub (Go).
  - mgpusim driver hook that exposes PMEngine API surface.
  - Python host scheduler that reads a KHT JSON and issues placement decisions and simulated/real transfers via PyTorch.
  - PyTorch fragment_migration_demo that demonstrates leases, migrations and lease-based eviction.

Quickstart (software prototype on your cluster):

1) Put Akita Go files
   - Copy `akita/flame/*.go` into your local Akita clone (e.g., ~/akita/flame/).
   - Update module import paths or `go.mod` replace directives if needed.

2) Put mgpusim driver hook
   - Copy `mgpusim/amd/driver/flame_integration.go` into mgpusim tree. Edit the import path to point to your local akita module.
   - Rebuild mgpusim samples if you want to integrate the PMEngine deeper.

3) Python environment
   - Ensure Python 3.9+, PyTorch with CUDA and GPUs available.
   - Place `host_scheduler.py`, `fragment_migration_demo.py`, and `kht_example.json` in a workspace dir.

4) Run the host scheduler in simulate mode (no real copies):
   $ python3 host_scheduler.py kht_example.json

   Run the host scheduler executing real peer copies (requires multiple GPUs and torch.cuda):
   $ python3 host_scheduler.py kht_example.json --execute

5) Run the demo:
   $ python3 fragment_migration_demo.py

Notes:
- To connect host_scheduler to a real PM-Engine in Akita/mgpusim, replace the `execute_transfers` function with an RPC call or a local socket to the driver which calls `PMEngineController.RequestFragmentTransfer`.
- The Go PMEngine is a simulation stub. To perform actual DMA in a real driver you would implement a kernel/driver that performs cudaMemcpyPeer or GPU-Direct RDMA; in Akita/MGPUSim you can model latency via the PMEngine stub to reflect NVLink bandwidth/latency.

Suggested next steps:
- Replace the simple greedy placement with the full utility formula (includes predicted interference, replication knapsack).
- Add a small REST/RPC server in the driver to accept placement commands from host_scheduler (or use gRPC).
- Add logging/tracing hooks into Akita's monitoring for fragment events and metrics.

If you want, I will:
- add a gRPC client/server for the host->driver PMEngine commands,
- port the KHT emitter to a small LLVM pass skeleton (or a static analyzer) that emits the JSON automatically for your LLM kernels,
- or convert the PMEngine stub to post events into Akita simulation engine (so Akita will generate simulated trace metrics).
