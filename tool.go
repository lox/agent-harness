package harness

import (
	"context"
	"encoding/json"
)

// ToolDef describes a tool's schema for the LLM.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Tool is a callable tool that the agent can use.
type Tool struct {
	ToolDef

	// Execute runs the tool with the given arguments.
	// The arguments are the raw JSON from the model's tool call.
	Execute func(ctx context.Context, call ToolCall) (*ToolResult, error)
}

// ToolResult is the output of a tool execution.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`

	// Content is sent back to the LLM as the tool's response.
	Content string `json:"content"`

	// UserContent is optionally shown to the user instead of/in addition to Content.
	// If empty, Content is used for both purposes.
	UserContent string `json:"user_content,omitempty"`

	// IsError indicates the tool execution failed.
	IsError bool `json:"is_error,omitempty"`

	// Metadata is arbitrary structured data for the caller (not sent to LLM).
	Metadata map[string]any `json:"metadata,omitempty"`
}
