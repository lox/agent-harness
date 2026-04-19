package conversation

import (
	"encoding/json"
	"fmt"

	harness "github.com/lox/agent-harness"
)

// Entry is a provider-neutral conversation unit with role-scoped parts.
type Entry struct {
	Role  harness.MessageRole
	Parts []Part
}

// PartKind identifies the kind of content carried by a conversation part.
type PartKind string

const (
	PartText       PartKind = "text"
	PartThinking   PartKind = "thinking"
	PartToolCall   PartKind = "tool_call"
	PartToolResult PartKind = "tool_result"
)

// Part is one block of content inside an Entry.
type Part struct {
	Kind       PartKind
	Text       string
	ToolCall   ToolCall
	ToolResult ToolResult
}

// ToolCall is a provider-neutral function call request.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResult is a provider-neutral function call result.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// EntriesFromChat converts harness chat params into provider-neutral entries.
func EntriesFromChat(system string, messages []harness.Message) ([]Entry, error) {
	out := make([]Entry, 0, len(messages)+1)
	if system != "" {
		out = append(out, Entry{
			Role: harness.RoleSystem,
			Parts: []Part{{
				Kind: PartText,
				Text: system,
			}},
		})
	}

	for _, msg := range messages {
		entry, ok, err := entryFromMessage(msg)
		if err != nil {
			return nil, err
		}
		if ok {
			out = append(out, entry)
		}
	}

	return out, nil
}

func entryFromMessage(msg harness.Message) (Entry, bool, error) {
	entry := Entry{Role: msg.Role}

	switch msg.Role {
	case harness.RoleSystem, harness.RoleUser:
		if msg.Content != "" {
			entry.Parts = append(entry.Parts, Part{Kind: PartText, Text: msg.Content})
		}
	case harness.RoleAssistant:
		if msg.Thinking != "" {
			entry.Parts = append(entry.Parts, Part{Kind: PartThinking, Text: msg.Thinking})
		}
		if msg.Content != "" {
			entry.Parts = append(entry.Parts, Part{Kind: PartText, Text: msg.Content})
		}
		for _, call := range msg.ToolCalls {
			entry.Parts = append(entry.Parts, Part{
				Kind: PartToolCall,
				ToolCall: ToolCall{
					ID:        call.ID,
					Name:      call.Name,
					Arguments: append(json.RawMessage(nil), call.Arguments...),
				},
			})
		}
	case harness.RoleTool:
		if msg.ToolResult == nil {
			return Entry{}, false, fmt.Errorf("tool message missing tool result")
		}
		entry.Parts = append(entry.Parts, Part{
			Kind: PartToolResult,
			ToolResult: ToolResult{
				ToolCallID: msg.ToolResult.ToolCallID,
				Content:    msg.ToolResult.Content,
				IsError:    msg.ToolResult.IsError,
			},
		})
	default:
		return Entry{}, false, fmt.Errorf("unsupported message role %q", msg.Role)
	}

	if len(entry.Parts) == 0 {
		return Entry{}, false, nil
	}

	return entry, true, nil
}
