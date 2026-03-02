package harness

import "context"

// Provider sends messages to an LLM and returns a response.
type Provider interface {
	// Chat sends the conversation to the model and returns the response.
	// tools may be nil if no tools are available for this call.
	Chat(ctx context.Context, params ChatParams) (*ChatResult, error)
}

// ChatParams contains everything needed for an LLM call.
type ChatParams struct {
	Model    string         // model name
	System   string         // system prompt
	Messages []Message      // conversation history
	Tools    []ToolDef      // available tool definitions
	Options  map[string]any // provider-specific options (temperature, max_tokens, etc.)

	// Optional: called with streaming deltas as they arrive.
	// If nil, the provider should still return the complete response.
	OnDelta func(Delta)
}

// ChatResult is what the provider returns.
type ChatResult struct {
	Message Message // the assistant's response (may include tool calls)
	Usage   *Usage  // token usage, if available
}

// Usage tracks token consumption.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Add adds usage values when provided.
func (u *Usage) Add(other *Usage) {
	if u == nil || other == nil {
		return
	}
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
}

// Delta represents a streaming chunk from the provider.
type Delta struct {
	Text     string         // text content delta
	Thinking string         // thinking/reasoning delta
	ToolCall *ToolCallDelta // tool call delta if present
}

// ToolCallDelta is an incremental tool-call update during streaming.
type ToolCallDelta struct {
	Index     int    // index of the tool call in the response
	ID        string // only set on first delta for this tool call
	Name      string // only set on first delta for this tool call
	Arguments string // JSON fragment
}
