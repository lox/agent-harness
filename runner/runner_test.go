package runner

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestStartAndCompletionCleanup(t *testing.T) {
	r := New()

	done, err := r.Start(context.Background(), "thread-1", func(ctx context.Context) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(10 * time.Millisecond):
			return nil
		}
	})
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if !r.IsRunning("thread-1") {
		t.Fatalf("expected running true after Start")
	}

	if err := <-done; err != nil {
		t.Fatalf("done error = %v", err)
	}
	if r.IsRunning("thread-1") {
		t.Fatalf("expected running false after completion")
	}
}

func TestStopCancelsRun(t *testing.T) {
	r := New()

	done, err := r.Start(context.Background(), "thread-2", func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	if ok := r.Stop("thread-2"); !ok {
		t.Fatalf("Stop() = false, want true")
	}

	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("done error = %v, want context.Canceled", err)
	}
}

func TestStartDuplicateRunID(t *testing.T) {
	r := New()

	_, err := r.Start(context.Background(), "thread-3", func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if err != nil {
		t.Fatalf("first Start() error = %v", err)
	}

	_, err = r.Start(context.Background(), "thread-3", func(ctx context.Context) error {
		return nil
	})
	if !errors.Is(err, ErrAlreadyRunning) {
		t.Fatalf("second Start() error = %v, want ErrAlreadyRunning", err)
	}
}

func TestStartValidation(t *testing.T) {
	r := New()

	if _, err := r.Start(context.Background(), "", func(ctx context.Context) error { return nil }); !errors.Is(err, ErrInvalidRunID) {
		t.Fatalf("Start(empty) error = %v, want ErrInvalidRunID", err)
	}
	if _, err := r.Start(context.Background(), "thread-4", nil); !errors.Is(err, ErrNilRunFunc) {
		t.Fatalf("Start(nil fn) error = %v, want ErrNilRunFunc", err)
	}
}

func TestStopUnknownRunID(t *testing.T) {
	r := New()
	if ok := r.Stop("missing"); ok {
		t.Fatalf("Stop(missing) = true, want false")
	}
}
