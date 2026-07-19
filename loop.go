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
	StopEndTurn    StopReason = iota // model finished naturally (no more tool calls)
	StopMaxSteps                     // step budget exhausted
	StopCancelled                    // context cancelled
	StopPaused                       // hook requested pause (e.g. for approval)
	StopRefusal                      // provider refused the request
	StopIncomplete                   // provider returned incomplete or token-limited output
	StopError                        // run failed after producing a partial result
)

// Result contains everything produced by a single Run invocation.
type Result struct {
	// Messages contains all messages generated during this run,
	// in order. Includes assistant messages and tool result messages.
	// When StopReason is StopPaused, the assistant message contains the proposed
	// calls but no tool result messages exist for the pending calls yet.
	Messages []Message `json:"messages"`

	// TotalUsage is the sum of all LLM calls made during this run.
	TotalUsage Usage `json:"total_usage"`

	// Steps is the number of LLM calls made (1 = no tool calls).
	Steps int `json:"steps"`

	// StopReason indicates why the loop terminated.
	StopReason StopReason `json:"stop_reason"`

	// FinishReason is the last provider finish reason observed. It preserves
	// distinctions that StopReason groups together, such as max_tokens and
	// incomplete.
	FinishReason FinishReason `json:"finish_reason,omitempty"`

	// FinishDetails contains optional provider detail for FinishReason, such as
	// a refusal explanation or the cause of incomplete output.
	FinishDetails string `json:"finish_details,omitempty"`

	// ResponseID is the most recent non-empty provider response identifier.
	ResponseID string `json:"response_id,omitempty"`

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
	fail := func(err error) (*Result, error) {
		result.StopReason = StopError
		return result, err
	}
	recordToolResult := func(step int, call ToolCall, toolResult *ToolResult) {
		stored := cloneToolResult(toolResult)
		stored.ToolCallID = call.ID
		toolMsg := Message{Role: RoleTool, ToolResult: stored}
		messages = append(messages, toolMsg)
		result.Messages = append(result.Messages, toolMsg)
		toolMsgEvent := toolMsg
		emit(cfg, Event{Type: EventMessage, Step: step, Message: &toolMsgEvent})
	}
	recordUnexecuted := func(step int, calls []ToolCall, content string) {
		for _, call := range calls {
			recordToolResult(step, call, &ToolResult{
				Content: content,
				IsError: true,
			})
		}
	}

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
			return fail(fmt.Errorf("step %d: invalid tool set: %w", step, err))
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
			return fail(fmt.Errorf("step %d: %w", step, err))
		}
		if chatResult == nil {
			return fail(fmt.Errorf("step %d: provider returned nil result", step))
		}

		assistantMsg := chatResult.Message
		finishReason, err := resolveFinishReason(chatResult.FinishReason, assistantMsg)
		if err != nil {
			return fail(fmt.Errorf("step %d: %w", step, err))
		}
		messages = append(messages, assistantMsg)
		result.Messages = append(result.Messages, assistantMsg)
		result.TotalUsage.Add(chatResult.Usage)
		result.Steps = step + 1
		if chatResult.ResponseID != "" {
			result.ResponseID = chatResult.ResponseID
		}

		result.FinishReason = finishReason
		result.FinishDetails = chatResult.FinishDetails

		assistantMsgEvent := assistantMsg
		emit(cfg, Event{Type: EventMessage, Step: step, Message: &assistantMsgEvent})
		emit(cfg, Event{Type: EventTurnEnd, Step: step})

		switch finishReason {
		case FinishReasonEndTurn:
			result.StopReason = StopEndTurn
			return result, nil
		case FinishReasonRefusal:
			result.StopReason = StopRefusal
			return result, nil
		case FinishReasonMaxTokens, FinishReasonIncomplete:
			result.StopReason = StopIncomplete
			return result, nil
		case FinishReasonContinuation:
			continue
		case FinishReasonToolUse:
			// Execute the calls below.
		}

		for i, call := range assistantMsg.ToolCalls {
			callEvent := call
			if cfg.beforeTool != nil {
				action, err := cfg.beforeTool(ctx, call)
				if err != nil {
					recordUnexecuted(step, assistantMsg.ToolCalls[i:], "Tool call not executed because the before-tool hook failed.")
					return fail(fmt.Errorf("before tool %q: %w", call.Name, err))
				}
				switch action {
				case ToolActionContinue:
				case ToolActionSkip:
					toolResult := &ToolResult{
						ToolCallID: call.ID,
						Content:    "Tool call skipped.",
						IsError:    true,
					}
					recordToolResult(step, call, toolResult)
					continue
				case ToolActionPause:
					result.StopReason = StopPaused
					result.PendingToolCalls = append([]ToolCall(nil), assistantMsg.ToolCalls[i:]...)
					return result, nil
				default:
					recordUnexecuted(step, assistantMsg.ToolCalls[i:], "Tool call not executed because the before-tool hook returned an invalid action.")
					return fail(fmt.Errorf("before tool %q returned invalid action %d", call.Name, action))
				}
			}

			emit(cfg, Event{Type: EventToolStart, Step: step, ToolCall: &callEvent})

			toolResult, toolErr := executeTool(ctx, toolMap, call)
			if toolErr != nil {
				if errors.Is(toolErr, context.Canceled) || errors.Is(toolErr, context.DeadlineExceeded) {
					recordUnexecuted(step, assistantMsg.ToolCalls[i:], "Tool execution cancelled.")
					result.StopReason = StopCancelled
					return result, nil
				}
				emit(cfg, Event{Type: EventError, Step: step, ToolCall: &callEvent, Error: toolErr, Result: toolResult})
			}

			if cfg.afterTool != nil {
				if err := cfg.afterTool(ctx, call, toolResult); err != nil {
					emit(cfg, Event{Type: EventToolEnd, Step: step, ToolCall: &callEvent, Result: toolResult})
					recordToolResult(step, call, toolResult)
					recordUnexecuted(step, assistantMsg.ToolCalls[i+1:], "Tool call not executed because the after-tool hook failed.")
					return fail(fmt.Errorf("after tool %q: %w", call.Name, err))
				}
			}

			emit(cfg, Event{Type: EventToolEnd, Step: step, ToolCall: &callEvent, Result: toolResult})

			recordToolResult(step, call, toolResult)
		}
	}

	result.StopReason = StopMaxSteps
	return result, nil
}

func resolveFinishReason(reason FinishReason, message Message) (FinishReason, error) {
	if reason == FinishReasonUnspecified {
		if len(message.ToolCalls) > 0 {
			return FinishReasonToolUse, nil
		}
		return FinishReasonEndTurn, nil
	}

	switch reason {
	case FinishReasonEndTurn:
		if len(message.ToolCalls) > 0 {
			return "", errors.New("provider returned end_turn with tool calls")
		}
	case FinishReasonToolUse:
		if len(message.ToolCalls) == 0 {
			return "", errors.New("provider returned tool_use without tool calls")
		}
	case FinishReasonContinuation:
		if len(message.ToolCalls) > 0 {
			return "", errors.New("provider returned continuation with tool calls")
		}
	case FinishReasonRefusal, FinishReasonMaxTokens, FinishReasonIncomplete:
		// These states are terminal even when a provider includes partial tool
		// calls. Run retains the message but never executes those calls.
	default:
		return "", fmt.Errorf("provider returned unknown finish reason %q", reason)
	}

	return reason, nil
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
	stored := cloneToolResult(res)
	stored.ToolCallID = call.ID
	return stored, nil
}
