// path: akita/flame/fragment_directory.go
// Lightweight, in-memory Fragment Directory for Akita simulation.
// This sits next to Akita's simulation components and provides a global mapping:
// (vpn,page_fragment_index) -> (node/gpu, phys_addr, size, flags)
// NOTE: this is a software-level directory useful for simulation / prototype.
package flame

import (
	"fmt"
	"sync"
)

// FragmentKey identifies a fragment by virtual page (VPN) and fragment index.
type FragmentKey struct {
	VPN   uint64 // virtual page number
	Index uint16 // sub-page fragment index (0..N-1)
}

// FragMapping is the mapping stored in the global directory for one fragment.
type FragMapping struct {
	NodeID    int    // node/gpu id
	PhysAddr  uint64 // simulated physical address (or unique id)
	Size      uint32 // bytes (e.g., 256, 512, 4096)
	Replica   bool   // whether this fragment is replicated
	LeaseEnds int64  // epoch or timestamp when lease expires (simulated)
	Flags     uint32 // custom flags (hot, write-heavy, reserved)
}

// FragmentDirectory is a concurrency-safe directory.
type FragmentDirectory struct {
	mu   sync.RWMutex
	data map[FragmentKey]FragMapping
}

// NewFragmentDirectory creates an empty fragment directory.
func NewFragmentDirectory() *FragmentDirectory {
	return &FragmentDirectory{
		data: make(map[FragmentKey]FragMapping),
	}
}

// Lookup returns the mapping and true if present.
func (d *FragmentDirectory) Lookup(vpn uint64, idx uint16) (FragMapping, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	fk := FragmentKey{VPN: vpn, Index: idx}
	m, ok := d.data[fk]
	return m, ok
}

// Install atomically installs/updates a mapping.
func (d *FragmentDirectory) Install(vpn uint64, idx uint16, m FragMapping) {
	d.mu.Lock()
	defer d.mu.Unlock()
	fk := FragmentKey{VPN: vpn, Index: idx}
	d.data[fk] = m
}

// Remove deletes a fragment mapping.
func (d *FragmentDirectory) Remove(vpn uint64, idx uint16) {
	d.mu.Lock()
	defer d.mu.Unlock()
	fk := FragmentKey{VPN: vpn, Index: idx}
	delete(d.data, fk)
}

// ScanForNode returns all fragments currently mapped to a node.
func (d *FragmentDirectory) ScanForNode(node int) map[FragmentKey]FragMapping {
	out := make(map[FragmentKey]FragMapping)
	d.mu.RLock()
	defer d.mu.RUnlock()
	for k, v := range d.data {
		if v.NodeID == node {
			out[k] = v
		}
	}
	return out
}

// DebugDump prints a compact snapshot (for logs).
func (d *FragmentDirectory) DebugDump() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	s := "FragmentDirectory Dump:\n"
	for k, v := range d.data {
		s += fmt.Sprintf("VPN=%#x idx=%d -> node=%d pa=%#x size=%d lease=%d flags=%#x\n",
			k.VPN, k.Index, v.NodeID, v.PhysAddr, v.Size, v.LeaseEnds, v.Flags)
	}
	return s
}
