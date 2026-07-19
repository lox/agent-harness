package harness

import (
	"context"
	"errors"
	"testing"
)

func TestThreadAppendAndResolvePending(t *testing.T) {
	th := NewThread()
	th.AddUser("hello")

	th.Append(&Result{
		Messages:         []Message{{Role: RoleAssistant, Content: "need approval"}},
		PendingToolCalls: []ToolCall{{ID: "1", Name: "write"}},
	})

	if len(th.PendingToolCalls) != 1 {
		t.Fatalf("len(PendingToolCalls) = %d, want 1", len(th.PendingToolCalls))
	}

	err := th.ResolvePending(context.Background(), func(_ context.Context, call ToolCall) (*ToolResult, error) {
		return &ToolResult{ToolCallID: "wrong", Content: "approved"}, nil
	})
	if err != nil {
		t.Fatalf("ResolvePending() error = %v", err)
	}

	if len(th.PendingToolCalls) != 0 {
		t.Fatalf("pending calls not cleared")
	}
	last := th.Messages[len(th.Messages)-1]
	if last.Role != RoleTool || last.ToolResult.Content != "approved" || last.ToolResult.ToolCallID != "1" {
		t.Fatalf("last message = %+v", last)
	}
}

func TestThreadResolvePendingErrorKeepsCompletedProgress(t *testing.T) {
	th := &Thread{PendingToolCalls: []ToolCall{{ID: "1", Name: "write"}, {ID: "2", Name: "write"}}}
	want := errors.New("nope")

	err := th.ResolvePending(context.Background(), func(_ context.Context, call ToolCall) (*ToolResult, error) {
		if call.ID == "2" {
			return nil, want
		}
		return &ToolResult{Content: "done"}, nil
	})
	if !errors.Is(err, want) {
		t.Fatalf("ResolvePending() error = %v, want %v", err, want)
	}
	if len(th.PendingToolCalls) != 1 || th.PendingToolCalls[0].ID != "2" {
		t.Fatalf("pending calls = %+v, want only call 2", th.PendingToolCalls)
	}
	if len(th.Messages) != 1 || th.Messages[0].ToolResult.ToolCallID != "1" {
		t.Fatalf("completed messages = %+v, want call 1 result", th.Messages)
	}
}

func TestThreadResolvePendingCopiesReusedResults(t *testing.T) {
	th := &Thread{PendingToolCalls: []ToolCall{{ID: "1", Name: "read"}, {ID: "2", Name: "read"}}}
	shared := &ToolResult{Content: "done"}
	if err := th.ResolvePending(context.Background(), func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return shared, nil
	}); err != nil {
		t.Fatalf("ResolvePending() error = %v", err)
	}
	if th.Messages[0].ToolResult.ToolCallID != "1" || th.Messages[1].ToolResult.ToolCallID != "2" {
		t.Fatalf("tool result IDs mutated: %+v", th.Messages)
	}
	if shared.ToolCallID != "" {
		t.Fatalf("resolver-owned result mutated: %+v", shared)
	}
}

func TestThreadResolvePendingNilResolverReturnsError(t *testing.T) {
	th := &Thread{PendingToolCalls: []ToolCall{{ID: "1", Name: "write"}}}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ResolvePending panicked with nil resolver: %v", r)
		}
	}()

	err := th.ResolvePending(context.Background(), nil)
	if err == nil {
		t.Fatalf("ResolvePending() error = nil, want non-nil")
	}
}
