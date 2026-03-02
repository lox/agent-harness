package harness

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
)

// Thread tracks conversation state between runs.
type Thread struct {
	ID               string         `json:"id"`
	Messages         []Message      `json:"messages"`
	PendingToolCalls []ToolCall     `json:"pending_tool_calls,omitempty"`
	Metadata         map[string]any `json:"metadata,omitempty"`
}

// NewThread creates a thread with a generated ID.
func NewThread() *Thread {
	return &Thread{ID: newThreadID()}
}

// AddUser appends a user message.
func (t *Thread) AddUser(content string) {
	t.Messages = append(t.Messages, Message{Role: RoleUser, Content: content})
}

// Append adds the messages from a run result and updates pending state.
func (t *Thread) Append(r *Result) {
	if r == nil {
		return
	}
	t.Messages = append(t.Messages, r.Messages...)
	t.PendingToolCalls = append([]ToolCall(nil), r.PendingToolCalls...)
}

// ResolvePending executes pending tool calls, appends the results as RoleTool
// messages, and clears pending state.
func (t *Thread) ResolvePending(ctx context.Context, fn func(ctx context.Context, call ToolCall) (*ToolResult, error)) error {
	if len(t.PendingToolCalls) == 0 {
		return nil
	}
	if fn == nil {
		return errors.New("resolve pending function cannot be nil")
	}

	resolved := make([]Message, 0, len(t.PendingToolCalls))
	for _, call := range t.PendingToolCalls {
		result, err := fn(ctx, call)
		if err != nil {
			return err
		}
		if result == nil {
			return errors.New("resolve pending returned nil result")
		}
		if result.ToolCallID == "" {
			result.ToolCallID = call.ID
		}
		resolved = append(resolved, Message{Role: RoleTool, ToolResult: result})
	}

	t.Messages = append(t.Messages, resolved...)
	t.PendingToolCalls = nil
	return nil
}

func newThreadID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		// In practice this should never fail on supported platforms.
		return "thread-unknown"
	}
	return "thread-" + hex.EncodeToString(b)
}
