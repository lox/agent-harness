package runner

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

var (
	ErrInvalidRunID   = errors.New("run id cannot be empty")
	ErrNilRunFunc     = errors.New("run function cannot be nil")
	ErrAlreadyRunning = errors.New("run already active")
)

type RunFunc func(ctx context.Context) error

type Runner struct {
	mu      sync.Mutex
	cancels map[string]context.CancelFunc
}

func New() *Runner {
	return &Runner{cancels: make(map[string]context.CancelFunc)}
}

func (r *Runner) Start(parent context.Context, runID string, fn RunFunc) (<-chan error, error) {
	if runID == "" {
		return nil, ErrInvalidRunID
	}
	if fn == nil {
		return nil, ErrNilRunFunc
	}

	r.mu.Lock()
	if _, exists := r.cancels[runID]; exists {
		r.mu.Unlock()
		return nil, fmt.Errorf("%w: %s", ErrAlreadyRunning, runID)
	}
	runCtx, cancel := context.WithCancel(parent)
	r.cancels[runID] = cancel
	r.mu.Unlock()

	done := make(chan error, 1)
	go func() {
		defer func() {
			r.mu.Lock()
			delete(r.cancels, runID)
			r.mu.Unlock()
			close(done)
		}()

		done <- fn(runCtx)
	}()

	return done, nil
}

func (r *Runner) Stop(runID string) bool {
	r.mu.Lock()
	cancel, ok := r.cancels[runID]
	r.mu.Unlock()
	if !ok {
		return false
	}
	cancel()
	return true
}

func (r *Runner) IsRunning(runID string) bool {
	r.mu.Lock()
	_, ok := r.cancels[runID]
	r.mu.Unlock()
	return ok
}
