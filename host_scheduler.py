# path: host_scheduler.py
# Host-side placement & transfer controller prototype.
# - Reads a KHT JSON file describing fragments (id, size, predicted reuse, importance, timescale)
# - Computes Ub,g and rho = Ub / cost and performs greedy placement under HBM/TLB budgets
# - Issues transfers by copying tensors with PyTorch (cuda peer copy) to simulate PM-Engine
# - Can run in 'simulate' mode (no copies) or 'execute' (real peer copies). 

import json
import math
import heapq
import argparse
import time
import torch
import os
from typing import Dict, List, Tuple

# KHT JSON format example:
# {
#   "fragments": [
#     {"id":"f0001","size":4096,"importance":0.9,"reuse":3,"timescale":"short"},
#     ...
#   ],
#   "nodes": [ {"id":0,"hbm_budget":8589934592,"tlb_budget":262144}, ... ]
# }

def load_kht(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def compute_cost(size_bytes: int, node: Dict):
    # cost model: bytes + tlb units conversion (1 tlb unit ~ 4KB)
    tlb_units = math.ceil(size_bytes / 4096)
    # cost in pseudo-bytes = size + tlb_units * 4096 * 0.1 (tlb pressure cost)
    return size_bytes + tlb_units * 4096 * 0.1

def compute_utility(frag: Dict, node: Dict):
    # Ub,g = w_r * reuse + w_i * importance - w_c * interference (use simple surrogate)
    w_r = 1.0
    w_i = 0.8
    w_c = 0.5
    interference = node.get('pred_interference', 0.0)
    ub = w_r * frag['reuse'] + w_i * frag['importance'] - w_c * interference
    return max(0.0, ub)

def plan_placements(kht: Dict, simulate_only: bool = True):
    fragments = kht['fragments']
    nodes = kht['nodes']
    # budgets: remaining hbm and tlb
    rem_hbm = {n['id']: n['hbm_budget'] for n in nodes}
    placements = {}  # frag_id -> node_id

    heap = []
    # build candidate list of (rho, frag_id, node_id)
    for f in fragments:
        for n in nodes:
            ub = compute_utility(f, n)
            cost = compute_cost(f['size'], n)
            if cost <= 0:
                continue
            rho = ub / cost
            # push negative rho so highest come first
            heapq.heappush(heap, (-rho, f['id'], n['id'], ub, cost))

    while heap:
        negrho, fid, nid, ub, cost = heapq.heappop(heap)
        if fid in placements:
            continue
        if rem_hbm[nid] >= cost:
            placements[fid] = nid
            rem_hbm[nid] -= cost
    return placements

# small utility to execute transfers by allocating small pinned tensors and doing .to()
def execute_transfers(placements: Dict[str,int], kht: Dict, simulate_only: bool=True):
    # create a small map of fragment id -> a torch tensor on source GPU
    # For prototype: assume fragments are currently on node 0
    src_node = 0
    frag_tensors = {}
    for f in kht['fragments']:
        fid = f['id']
        size = f['size']
        # allocate a byte tensor sized to the fragment (use float32 elements)
        n_elems = max(1, size // 4)
        t = torch.empty(n_elems, dtype=torch.float32, device=f'cuda:{src_node}')
        frag_tensors[fid] = t

    results = {}
    for fid, dst in placements.items():
        dst_dev = f'cuda:{dst}'
        size = next(f['size'] for f in kht['fragments'] if f['id']==fid)
        t_src = frag_tensors[fid]
        if simulate_only:
            # just sleep to model latency
            simulated_latency = 0.001 + (size / (1<<20)) * 0.001
            time.sleep(simulated_latency)
            results[fid] = {'dst':dst, 'latency':simulated_latency}
        else:
            # real peer copy (uses cudaMemcpyPeer under the hood)
            torch.cuda.synchronize()
            start = time.time()
            t_dst = t_src.to(dst_dev, non_blocking=True)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            results[fid] = {'dst':dst, 'latency':elapsed}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kht", type=str, help="KHT json path")
    parser.add_argument("--execute", action="store_true", help="Actually perform peer copies (requires GPUs)")
    args = parser.parse_args()
    kht = load_kht(args.kht)
    print("Planning placements ...")
    placements = plan_placements(kht, simulate_only=not args.execute)
    print("Placements:", placements)
    print("Executing transfers ... (simulate_only={!s})".format(not args.execute))
    res = execute_transfers(placements, kht, simulate_only=not args.execute)
    print("Transfer results:", res)

if __name__ == "__main__":
    main()
