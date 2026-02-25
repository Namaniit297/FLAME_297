// path: akita/flame/pm_engine.go
// PMEngine: prioritized DMA / RDMA executor stub for Akita simulation.
// In a full hardware design the PM-Engine would be a dedicated low-latency DMA/RDMA engine
// that performs prioritized transfers, TLB prefetches and emits completion events.
// Here we provide an API to enqueue prioritized transfers that other Akita components can call.
// This stub uses Akita's sim.Engine messaging primitives (TODO: integrate with actual Akita objects).

package flame

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// TransferRequest describes a prioritized fragment transfer.
type TransferRequest struct {
	ID        string
	SrcNode   int
	DstNode   int
	SizeBytes uint64
	Priority  int    // lower => higher priority
	Meta      string // optional metadata (e.g., fragment key)
	Done      chan error
	// Deadline, lease info etc can be added here.
}

// PMEngine simulates a prioritized DMA engine. It processes requests in priority order.
type PMEngine struct {
	mu       sync.Mutex
	pending  []*TransferRequest
	active   bool
	quit     chan struct{}
	interval time.Duration // per-transfer simulated latency base
}

// NewPMEngine creates a new stub PMEngine.
func NewPMEngine() *PMEngine {
	e := &PMEngine{
		pending:  make([]*TransferRequest, 0),
		quit:     make(chan struct{}),
		interval: 2 * time.Millisecond, // default simulated per-request base latency; tune in sim
	}
	go e.loop()
	return e
}

// EnqueueTransfer enqueues a transfer. Returns channel to wait for completion.
func (e *PMEngine) EnqueueTransfer(req *TransferRequest) (chan error, error) {
	if req == nil {
		return nil, errors.New("nil request")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	req.Done = make(chan error, 1)
	e.pending = append(e.pending, req)
	// keep pending sorted by Priority (simple insertion sort for small lists)
	for i := len(e.pending) - 1; i > 0; i-- {
		if e.pending[i].Priority < e.pending[i-1].Priority {
			e.pending[i], e.pending[i-1] = e.pending[i-1], e.pending[i]
		} else {
			break
		}
	}
	return req.Done, nil
}

// Stop stops the PMEngine loop.
func (e *PMEngine) Stop() {
	close(e.quit)
}

// loop simulates executing transfers one-by-one (priority order).
func (e *PMEngine) loop() {
	for {
		select {
		case <-e.quit:
			return
		default:
		}
		e.mu.Lock()
		if len(e.pending) == 0 {
			e.mu.Unlock()
			time.Sleep(1 * time.Millisecond)
			continue
		}
		req := e.pending[0]
		e.pending = e.pending[1:]
		e.mu.Unlock()

		// Simulate servicing: base latency + size-dependent delay
		latency := e.interval + time.Duration(req.SizeBytes/ (1<<20)) * 1*time.Millisecond
		// faster if high priority: subtract tiny amount
		if req.Priority <= 0 {
			latency /= 2
		}

		// If there were real Akita components, we'd post events on the sim engine.
		// For now we simulate with a goroutine and call Done when complete.
		go func(r *TransferRequest, l time.Duration) {
			time.Sleep(l)
			// mark done
			select {
			case r.Done <- nil:
			default:
			}
		}(req, latency)
	}
}

// SubmitWithContext provides a convenience function that waits until completion or context cancel.
func (e *PMEngine) SubmitWithContext(ctx context.Context, req *TransferRequest) error {
	done, err := e.EnqueueTransfer(req)
	if err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return err
	}
}

// Debug: simple stats
func (e *PMEngine) Stats() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return fmt.Sprintf("PMEngine: pending=%d", len(e.pending))
}
