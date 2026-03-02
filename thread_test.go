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
		return &ToolResult{ToolCallID: call.ID, Content: "approved"}, nil
	})
	if err != nil {
		t.Fatalf("ResolvePending() error = %v", err)
	}

	if len(th.PendingToolCalls) != 0 {
		t.Fatalf("pending calls not cleared")
	}
	last := th.Messages[len(th.Messages)-1]
	if last.Role != RoleTool || last.ToolResult.Content != "approved" {
		t.Fatalf("last message = %+v", last)
	}
}

func TestThreadResolvePendingErrorKeepsPending(t *testing.T) {
	th := &Thread{PendingToolCalls: []ToolCall{{ID: "1", Name: "write"}}}
	want := errors.New("nope")

	err := th.ResolvePending(context.Background(), func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return nil, want
	})
	if !errors.Is(err, want) {
		t.Fatalf("ResolvePending() error = %v, want %v", err, want)
	}
	if len(th.PendingToolCalls) != 1 {
		t.Fatalf("pending calls changed on error")
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
