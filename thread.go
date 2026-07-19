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

// ResolvePending executes pending tool calls and appends each successful result
// as a RoleTool message. If a later call fails, completed calls stay recorded
// and only the unexecuted calls remain pending.
func (t *Thread) ResolvePending(ctx context.Context, fn func(ctx context.Context, call ToolCall) (*ToolResult, error)) error {
	if len(t.PendingToolCalls) == 0 {
		return nil
	}
	if fn == nil {
		return errors.New("resolve pending function cannot be nil")
	}

	for len(t.PendingToolCalls) > 0 {
		call := t.PendingToolCalls[0]
		result, err := fn(ctx, call)
		if err != nil {
			return err
		}
		if result == nil {
			return errors.New("resolve pending returned nil result")
		}
		stored := cloneToolResult(result)
		stored.ToolCallID = call.ID
		t.Messages = append(t.Messages, Message{Role: RoleTool, ToolResult: stored})
		t.PendingToolCalls = t.PendingToolCalls[1:]
	}
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
