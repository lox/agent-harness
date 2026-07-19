package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	harness "github.com/lox/agent-harness"
	mem "github.com/lox/agent-harness/memory"
	"github.com/lox/agent-harness/runner"
)

func TestPrintResultSuppressesDuplicateAssistantOutput(t *testing.T) {
	a := &app{}

	out := captureStdout(t, func() {
		a.printResult(&harness.Result{Messages: []harness.Message{
			{Role: harness.RoleTool, ToolResult: &harness.ToolResult{UserContent: "same output"}},
			{Role: harness.RoleAssistant, Content: "same output"},
		}})
	})

	if !strings.Contains(out, "tool      │ same output") {
		t.Fatalf("expected tool output in %q", out)
	}
	if strings.Contains(out, "assistant │ same output") {
		t.Fatalf("expected duplicate assistant output to be suppressed, got %q", out)
	}
}

func TestHandleCommandStopCancelsActiveRun(t *testing.T) {
	r := runner.New()
	th := harness.NewThread()
	a := &app{runner: r, thread: th}

	done, err := r.Start(context.Background(), th.ID, func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if err != nil {
		t.Fatalf("start run: %v", err)
	}

	out := captureStdout(t, func() {
		handled, quit := a.handleCommand("/stop")
		if !handled || quit {
			t.Fatalf("unexpected command result handled=%v quit=%v", handled, quit)
		}
	})

	if !strings.Contains(out, "cancelled active run") {
		t.Fatalf("expected cancel message, got %q", out)
	}
	if runErr := <-done; runErr == nil {
		t.Fatalf("expected cancelled run error")
	}
}

func TestHandleCommandRememberWritesMemory(t *testing.T) {
	store, err := mem.New(t.TempDir(), mem.WithClock(func() time.Time {
		return time.Date(2026, 5, 23, 12, 0, 0, 0, time.UTC)
	}))
	if err != nil {
		t.Fatalf("memory.New() error = %v", err)
	}
	a := &app{memory: store}

	out := captureStdout(t, func() {
		handled, quit := a.handleCommand("/remember Prefer direct answers")
		if !handled || quit {
			t.Fatalf("unexpected command result handled=%v quit=%v", handled, quit)
		}
	})
	if !strings.Contains(out, "wrote memory/2026-05-23.md") {
		t.Fatalf("expected memory write output, got %q", out)
	}

	content, err := os.ReadFile(storeFile(t, store, "memory/2026-05-23.md"))
	if err != nil {
		t.Fatalf("read memory file: %v", err)
	}
	if !strings.Contains(string(content), "Prefer direct answers") {
		t.Fatalf("memory content = %s", content)
	}
}

func TestHandleCommandRememberWithoutTextShowsUsage(t *testing.T) {
	store, err := mem.New(t.TempDir())
	if err != nil {
		t.Fatalf("memory.New() error = %v", err)
	}
	a := &app{memory: store}

	out := captureStdout(t, func() {
		handled, quit := a.handleCommand("/remember")
		if !handled || quit {
			t.Fatalf("unexpected command result handled=%v quit=%v", handled, quit)
		}
	})
	if !strings.Contains(out, "usage: /remember <text>") {
		t.Fatalf("expected remember usage, got %q", out)
	}
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w

	fn()

	_ = w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("copy stdout: %v", err)
	}
	_ = r.Close()

	return buf.String()
}

func storeFile(t *testing.T, store *mem.Store, relPath string) string {
	t.Helper()

	return filepath.Join(store.WorkspaceDir(), filepath.FromSlash(relPath))
}
