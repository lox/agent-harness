package harness

import "encoding/json"

// MessageRole identifies the role of a message in the conversation.
type MessageRole string

const (
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleSystem    MessageRole = "system"
	RoleTool      MessageRole = "tool"
)

// Message is a single message in the conversation history.
// Each message has exactly one of: text content, tool calls, or a tool result.
type Message struct {
	Role    MessageRole `json:"role"`
	Content string      `json:"content,omitempty"`

	// For RoleAssistant messages that include tool calls.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// For RoleTool messages — the result of a single tool call.
	ToolResult *ToolResult `json:"tool_result,omitempty"`

	// Thinking/reasoning content from the model (if any).
	Thinking string `json:"thinking,omitempty"`
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}
