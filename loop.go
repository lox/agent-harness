package harness

import (
	"context"
	"errors"
	"fmt"
	"sort"
)

// StopReason indicates why a run terminated.
type StopReason int

const (
	StopEndTurn   StopReason = iota // model finished naturally (no more tool calls)
	StopMaxSteps                    // step budget exhausted
	StopCancelled                   // context cancelled
	StopPaused                      // hook requested pause (e.g. for approval)
)

// Result contains everything produced by a single Run invocation.
type Result struct {
	// Messages contains all messages generated during this run,
	// in order. Includes assistant messages and tool result messages.
	// When StopReason is StopPaused, these are the messages completed
	// before the pause — they do NOT include the pending tool calls.
	Messages []Message `json:"messages"`

	// TotalUsage is the sum of all LLM calls made during this run.
	TotalUsage Usage `json:"total_usage"`

	// Steps is the number of LLM calls made (1 = no tool calls).
	Steps int `json:"steps"`

	// StopReason indicates why the loop terminated.
	StopReason StopReason `json:"stop_reason"`

	// PendingToolCalls contains tool calls that were proposed by the model
	// but not yet executed when the loop was paused.
	PendingToolCalls []ToolCall `json:"pending_tool_calls,omitempty"`
}

// Run executes the agent loop: call the LLM, execute any tool calls,
// feed results back, repeat until the model stops calling tools or
// the step budget is exhausted.
func Run(ctx context.Context, provider Provider, opts ...Option) (*Result, error) {
	if provider == nil {
		return nil, errors.New("provider is required")
	}

	cfg := applyOptions(opts...)
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	messages := append([]Message(nil), cfg.messages...)
	activeTools := stableTools(cfg.tools)
	toolMap := buildToolMap(activeTools)
	result := &Result{}

	for step := 0; step < cfg.maxSteps; step++ {
		if ctx.Err() != nil {
			result.StopReason = StopCancelled
			return result, nil
		}

		emit(cfg, Event{Type: EventTurnStart, Step: step})

		if cfg.toolFilter != nil {
			activeTools = cfg.toolFilter(step, append([]Message(nil), messages...))
		} else {
			activeTools = stableTools(cfg.tools)
		}
		if err := validateTools(activeTools); err != nil {
			return nil, fmt.Errorf("step %d: invalid tool set: %w", step, err)
		}
		toolMap = buildToolMap(activeTools)

		chatResult, err := provider.Chat(ctx, ChatParams{
			Model:    cfg.model,
			System:   cfg.system,
			Messages: append([]Message(nil), messages...),
			Tools:    toolDefs(activeTools),
			Options:  copyMap(cfg.providerOpts),
			OnDelta:  cfg.onDelta,
		})
		if err != nil {
			if ctx.Err() != nil {
				result.StopReason = StopCancelled
				return result, nil
			}
			return nil, fmt.Errorf("step %d: %w", step, err)
		}
		if chatResult == nil {
			return nil, fmt.Errorf("step %d: provider returned nil result", step)
		}

		assistantMsg := chatResult.Message
		messages = append(messages, assistantMsg)
		result.Messages = append(result.Messages, assistantMsg)
		result.TotalUsage.Add(chatResult.Usage)
		result.Steps = step + 1

		assistantMsgEvent := assistantMsg
		emit(cfg, Event{Type: EventMessage, Step: step, Message: &assistantMsgEvent})
		emit(cfg, Event{Type: EventTurnEnd, Step: step})

		if len(assistantMsg.ToolCalls) == 0 {
			result.StopReason = StopEndTurn
			return result, nil
		}

		for i, call := range assistantMsg.ToolCalls {
			callEvent := call
			if cfg.beforeTool != nil {
				action, err := cfg.beforeTool(ctx, call)
				if err != nil {
					return nil, err
				}
				switch action {
				case ToolActionSkip:
					toolResult := &ToolResult{
						ToolCallID: call.ID,
						Content:    "Tool call skipped.",
						IsError:    true,
					}
					toolMsg := Message{Role: RoleTool, ToolResult: toolResult}
					messages = append(messages, toolMsg)
					result.Messages = append(result.Messages, toolMsg)
					toolMsgEvent := toolMsg
					emit(cfg, Event{Type: EventMessage, Step: step, Message: &toolMsgEvent})
					continue
				case ToolActionPause:
					result.StopReason = StopPaused
					result.PendingToolCalls = append([]ToolCall(nil), assistantMsg.ToolCalls[i:]...)
					return result, nil
				}
			}

			emit(cfg, Event{Type: EventToolStart, Step: step, ToolCall: &callEvent})

			toolResult, toolErr := executeTool(ctx, toolMap, call)
			if toolErr != nil {
				if errors.Is(toolErr, context.Canceled) || errors.Is(toolErr, context.DeadlineExceeded) {
					result.StopReason = StopCancelled
					return result, nil
				}
				emit(cfg, Event{Type: EventError, Step: step, ToolCall: &callEvent, Error: toolErr, Result: toolResult})
			}

			if cfg.afterTool != nil {
				if err := cfg.afterTool(ctx, call, toolResult); err != nil {
					return nil, err
				}
			}

			emit(cfg, Event{Type: EventToolEnd, Step: step, ToolCall: &callEvent, Result: toolResult})

			toolMsg := Message{Role: RoleTool, ToolResult: toolResult}
			messages = append(messages, toolMsg)
			result.Messages = append(result.Messages, toolMsg)
			toolMsgEvent := toolMsg
			emit(cfg, Event{Type: EventMessage, Step: step, Message: &toolMsgEvent})
		}
	}

	result.StopReason = StopMaxSteps
	return result, nil
}

func buildToolMap(tools []Tool) map[string]Tool {
	m := make(map[string]Tool, len(tools))
	for _, tool := range tools {
		m[tool.Name] = tool
	}
	return m
}

func toolDefs(tools []Tool) []ToolDef {
	if len(tools) == 0 {
		return nil
	}
	defs := make([]ToolDef, 0, len(tools))
	for _, tool := range tools {
		defs = append(defs, tool.ToolDef)
	}
	return defs
}

func stableTools(tools []Tool) []Tool {
	out := append([]Tool(nil), tools...)
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

func executeTool(ctx context.Context, toolMap map[string]Tool, call ToolCall) (*ToolResult, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	tool, ok := toolMap[call.Name]
	if !ok {
		return &ToolResult{
			ToolCallID: call.ID,
			Content:    fmt.Sprintf("Unknown tool: %s", call.Name),
			IsError:    true,
		}, fmt.Errorf("unknown tool %q", call.Name)
	}

	res, err := tool.Execute(ctx, call)
	if err != nil {
		return &ToolResult{
			ToolCallID: call.ID,
			Content:    "Tool execution failed.",
			IsError:    true,
			Metadata:   map[string]any{"error": err.Error()},
		}, err
	}
	if res == nil {
		return &ToolResult{
			ToolCallID: call.ID,
			Content:    "Tool execution failed.",
			IsError:    true,
		}, errors.New("tool returned nil result")
	}
	if res.ToolCallID == "" {
		res.ToolCallID = call.ID
	}
	return res, nil
}
