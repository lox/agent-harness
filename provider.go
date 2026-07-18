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
	Message       Message      // the assistant's response (may include tool calls)
	ResponseID    string       // provider-assigned response identifier, if available
	FinishReason  FinishReason // why the provider ended this response
	FinishDetails string       // optional provider detail about the finish reason
	Usage         *Usage       // token usage, if available
}

// FinishReason describes why a provider response ended. Providers should set a
// reason explicitly. An unspecified reason is accepted for compatibility and
// inferred as end_turn or tool_use from the returned message.
type FinishReason string

const (
	FinishReasonUnspecified  FinishReason = ""
	FinishReasonEndTurn      FinishReason = "end_turn"
	FinishReasonToolUse      FinishReason = "tool_use"
	FinishReasonRefusal      FinishReason = "refusal"
	FinishReasonMaxTokens    FinishReason = "max_tokens"
	FinishReasonIncomplete   FinishReason = "incomplete"
	FinishReasonContinuation FinishReason = "continuation"
)

// Usage tracks token consumption.
type Usage struct {
	InputTokens                int `json:"input_tokens"`
	OutputTokens               int `json:"output_tokens"`
	CachedInputTokens          int `json:"cached_input_tokens,omitempty"`
	CacheCreationInputTokens   int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens       int `json:"cache_read_input_tokens,omitempty"`
	CacheCreation5mInputTokens int `json:"cache_creation_5m_input_tokens,omitempty"`
	CacheCreation1hInputTokens int `json:"cache_creation_1h_input_tokens,omitempty"`
}

// Add adds usage values when provided.
func (u *Usage) Add(other *Usage) {
	if u == nil || other == nil {
		return
	}
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.CachedInputTokens += other.CachedInputTokens
	u.CacheCreationInputTokens += other.CacheCreationInputTokens
	u.CacheReadInputTokens += other.CacheReadInputTokens
	u.CacheCreation5mInputTokens += other.CacheCreation5mInputTokens
	u.CacheCreation1hInputTokens += other.CacheCreation1hInputTokens
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
