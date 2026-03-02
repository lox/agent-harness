package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"strings"
	"testing"

	harness "github.com/lox/agent-harness"
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
