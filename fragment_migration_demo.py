# path: fragment_migration_demo.py
# Demo: maintain a set of simulated fragments on GPUs, give each a lease and simple access pattern,
# then run lease-based eviction + renewal. This demonstrates the runtime mechanics of FLAME/DynKV.
# REQUIREMENTS: PyTorch with CUDA and at least 2 visible GPUs for true peer copy timing.

import torch
import time
import random
from collections import defaultdict, deque

# Simulation parameters
N_FRAGMENTS = 64
FRAG_SIZE = 4096        # bytes
SRC_GPU = 0
GPUS = list(range(torch.cuda.device_count()))
if len(GPUS) < 1:
    raise SystemExit("No GPUs found; run on a multi-GPU node for full demo.")
print("GPUs available:", GPUS)
# create fragments on SRC_GPU
fragments = {}
for i in range(N_FRAGMENTS):
    n_elems = max(1, FRAG_SIZE // 4)
    fragments[f"f{i:04d}"] = torch.randn(n_elems, device=f"cuda:{SRC_GPU}")

# Simulated leases: frag_id -> expiry_epoch (integer)
epoch = 0
leases = {fid: epoch + random.randint(1,5) for fid in fragments}
# Simulated hotness counters
hotness = {fid: random.random() for fid in fragments}
# Residency map: frag -> node
residency = {fid: SRC_GPU for fid in fragments}

def access_pattern_step():
    # sample a few fragments with bias on hotness
    sample = random.choices(list(fragments.keys()), weights=[hotness[f] for f in fragments], k=8)
    return sample

def migrate_fragment(fid, dst):
    src = residency[fid]
    if src == dst:
        return 0.0
    t_src = fragments[fid].to(f"cuda:{dst}", non_blocking=True)
    torch.cuda.synchronize()
    residency[fid] = dst
    return 0.0

def lease_eviction(epoch):
    # evict fragments whose lease expired and not hot
    evicted = []
    for fid, expiry in list(leases.items()):
        if expiry <= epoch and hotness[fid] < 0.5:
            # migrate back to SRC_GPU (or host) to free local HBM
            migrate_fragment(fid, SRC_GPU)
            # extend lease small amount if observed hotness rises
            leases[fid] = epoch + 1
            evicted.append(fid)
    return evicted

print("Starting token-generation-style simulation...")
for ep in range(1, 101):
    epoch = ep
    touches = access_pattern_step()
    # simulate accesses; record observed reuse and bump hotness
    for fid in touches:
        # the actual kernel would use the fragment on the GPU where it resides
        device = residency[fid]
        # fake compute
        _ = fragments[fid].sum().item()
        hotness[fid] += 0.01
        # on access we might extend lease
        leases[fid] = max(leases[fid], epoch + 2)

    # every 5 epochs, decide placement for top-K hotness fragments and migrate to GPU 1 (if exists)
    if ep % 5 == 0 and len(GPUS) > 1:
        # pick top-4 hot fragments and migrate to GPU1
        topk = sorted(hotness.items(), key=lambda kv: -kv[1])[:4]
        for fid, _ in topk:
            t0 = time.time()
            migrate_fragment(fid, GPUS[1])
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"Epoch {ep}: migrated {fid} to GPU {GPUS[1]} took {t1-t0:.6f}s")

    # do lease eviction
    evicted = lease_eviction(epoch)
    if evicted:
        print(f"Epoch {ep}: evicted {len(evicted)} fragments (sample {evicted[:3]})")

    time.sleep(0.02)

print("Demo finished.")
