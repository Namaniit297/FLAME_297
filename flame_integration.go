// path: mgpusim/amd/driver/flame_integration.go
// Integration glue: expose a PM-Engine handle to the mgpusim driver and a simple command
// to request prioritized fragment transfer (used by the Python host-scheduler via RPC or CLI).
// Minimal, safe changes that compile when placed in mgpusim/amd/driver package.

package driver

import (
	"context"
	"fmt"
	"time"

	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/akita/v4/clock"
	"github.com/sarchlab/akita/v4/device"
	"github.com/sarchlab/akita/v4/tracing"

	// local flame package import assumes akita/flame is accessible
	"github.com/your-local-path/akita/flame" // MODIFY this import path to your local layout
)

// NOTE: adjust import path above to your workspace (replace github.com/your-local-path with real path).

// PMEngineController is embedded into Driver to access PMEngine from mgpusim driver APIs.
type PMEngineController struct {
	Engine *flame.PMEngine
}

// NewPMEngineController creates a PMEngineController.
func NewPMEngineController() *PMEngineController {
	return &PMEngineController{
		Engine: flame.NewPMEngine(),
	}
}

// RequestFragmentTransfer enqueues a transfer on the PMEngine and returns when complete.
func (p *PMEngineController) RequestFragmentTransfer(ctx context.Context, srcGPU, dstGPU int, sizeBytes uint64, priority int, meta string) error {
	req := &flame.TransferRequest{
		ID:        fmt.Sprintf("tx-%d-%d-%d", srcGPU, dstGPU, time.Now().UnixNano()),
		SrcNode:   srcGPU,
		DstNode:   dstGPU,
		SizeBytes: sizeBytes,
		Priority:  priority,
		Meta:      meta,
	}
	// Submit and wait until done or ctx canceled.
	return p.Engine.SubmitWithContext(ctx, req)
}

// Hook: example of how Driver may call this (call from higher-level host scheduler).
// In your driver struct, add a PMEngineController field and route commands from host.
