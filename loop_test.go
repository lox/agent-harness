package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

type mockProvider struct {
	results []*ChatResult
	err     error
	calls   []ChatParams
	chat    func(context.Context, ChatParams) (*ChatResult, error)
}

func (m *mockProvider) Chat(ctx context.Context, params ChatParams) (*ChatResult, error) {
	m.calls = append(m.calls, params)
	if m.chat != nil {
		return m.chat(ctx, params)
	}
	if m.err != nil {
		return nil, m.err
	}
	if len(m.results) == 0 {
		return &ChatResult{}, nil
	}
	r := m.results[0]
	m.results = m.results[1:]
	return r, nil
}

func TestRunEndTurnWithoutTools(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message:      Message{Role: RoleAssistant, Content: "done"},
		ResponseID:   "response-1",
		FinishReason: FinishReasonEndTurn,
		Usage:        &Usage{InputTokens: 3, OutputTokens: 5},
	}}}

	res, err := Run(context.Background(), p)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if res.StopReason != StopEndTurn {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopEndTurn)
	}
	if res.Steps != 1 {
		t.Fatalf("Steps = %d, want 1", res.Steps)
	}
	if res.FinishReason != FinishReasonEndTurn {
		t.Fatalf("FinishReason = %q, want %q", res.FinishReason, FinishReasonEndTurn)
	}
	if res.ResponseID != "response-1" {
		t.Fatalf("ResponseID = %q, want response-1", res.ResponseID)
	}
	if got := res.TotalUsage; got.InputTokens != 3 || got.OutputTokens != 5 {
		t.Fatalf("TotalUsage = %+v", got)
	}
	if len(res.Messages) != 1 || res.Messages[0].Content != "done" {
		t.Fatalf("Messages = %+v", res.Messages)
	}
}

func TestRunExecutesToolCalls(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{
			Message:      Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
			FinishReason: FinishReasonToolUse,
		},
		{Message: Message{Role: RoleAssistant, Content: "final"}, FinishReason: FinishReasonEndTurn},
	}}

	tool := Tool{
		ToolDef: ToolDef{Name: "echo"},
		Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
			return &ToolResult{ToolCallID: call.ID, Content: "echoed"}, nil
		},
	}

	res, err := Run(context.Background(), p, WithTools(tool))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if res.StopReason != StopEndTurn {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopEndTurn)
	}
	if len(res.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3", len(res.Messages))
	}
	if res.Messages[1].Role != RoleTool || res.Messages[1].ToolResult.Content != "echoed" {
		t.Fatalf("tool result message = %+v", res.Messages[1])
	}
}

func TestRunStopsOnRefusal(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message:       Message{Role: RoleAssistant, Content: "I can't help with that."},
		ResponseID:    "response-refusal",
		FinishReason:  FinishReasonRefusal,
		FinishDetails: "safety: disallowed content",
	}}}

	res, err := Run(context.Background(), p)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopRefusal {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopRefusal)
	}
	if res.FinishReason != FinishReasonRefusal {
		t.Fatalf("FinishReason = %q, want %q", res.FinishReason, FinishReasonRefusal)
	}
	if res.FinishDetails != "safety: disallowed content" {
		t.Fatalf("FinishDetails = %q", res.FinishDetails)
	}
	if len(res.Messages) != 1 || res.Messages[0].Content == "" {
		t.Fatalf("Messages = %+v, want retained refusal", res.Messages)
	}
}

func TestRunStopsOnIncompleteOutput(t *testing.T) {
	tests := []struct {
		name   string
		reason FinishReason
	}{
		{name: "max tokens", reason: FinishReasonMaxTokens},
		{name: "other incomplete output", reason: FinishReasonIncomplete},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &mockProvider{results: []*ChatResult{{
				Message:      Message{Role: RoleAssistant, Content: "partial"},
				FinishReason: tt.reason,
			}}}

			res, err := Run(context.Background(), p)
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if res.StopReason != StopIncomplete {
				t.Fatalf("StopReason = %v, want %v", res.StopReason, StopIncomplete)
			}
			if res.FinishReason != tt.reason {
				t.Fatalf("FinishReason = %q, want %q", res.FinishReason, tt.reason)
			}
		})
	}
}

func TestRunContinuesWhenProviderRequestsContinuation(t *testing.T) {
	providerData := json.RawMessage(`{"type":"pause_turn","opaque":"signed-state"}`)
	p := &mockProvider{results: []*ChatResult{
		{
			Message:      Message{Role: RoleAssistant, Content: "provider paused", ProviderData: providerData},
			ResponseID:   "response-pause",
			FinishReason: FinishReasonContinuation,
		},
		{
			Message:      Message{Role: RoleAssistant, Content: "done"},
			FinishReason: FinishReasonEndTurn,
		},
	}}

	res, err := Run(context.Background(), p, WithPreviousResponseID("response-before-run"))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopEndTurn {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopEndTurn)
	}
	if res.Steps != 2 || len(p.calls) != 2 {
		t.Fatalf("Steps = %d, provider calls = %d, want 2", res.Steps, len(p.calls))
	}
	if len(res.Messages) != 2 || len(p.calls[1].Messages) != 1 {
		t.Fatalf("continuation history was not retained: result=%+v call=%+v", res.Messages, p.calls[1].Messages)
	}
	if !reflect.DeepEqual(p.calls[1].Messages[0].ProviderData, providerData) {
		t.Fatalf("ProviderData = %s, want %s", p.calls[1].Messages[0].ProviderData, providerData)
	}
	if res.ResponseID != "response-pause" {
		t.Fatalf("ResponseID = %q, want latest non-empty response-pause", res.ResponseID)
	}
	if res.FinishReason != FinishReasonEndTurn {
		t.Fatalf("FinishReason = %q, want %q", res.FinishReason, FinishReasonEndTurn)
	}
	if p.calls[0].PreviousResponseID != "response-before-run" || p.calls[1].PreviousResponseID != "" {
		t.Fatalf("PreviousResponseID by call = %q, %q", p.calls[0].PreviousResponseID, p.calls[1].PreviousResponseID)
	}
	if len(res.CallUsage) != 2 || res.CallUsage[0] != (Usage{}) || res.CallUsage[1] != (Usage{}) {
		t.Fatalf("CallUsage = %+v, want two zero values for omitted usage", res.CallUsage)
	}
}

func TestRunRetainsAndAggregatesCacheAwareUsage(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{
			Message:      Message{Role: RoleAssistant, Content: "continuing"},
			FinishReason: FinishReasonContinuation,
			Usage: &Usage{
				InputTokens:                10,
				OutputTokens:               2,
				CachedInputTokens:          3,
				CacheCreationInputTokens:   4,
				CacheReadInputTokens:       5,
				CacheCreation5mInputTokens: 6,
				CacheCreation1hInputTokens: 7,
			},
		},
		{
			Message:      Message{Role: RoleAssistant, Content: "done"},
			FinishReason: FinishReasonEndTurn,
			Usage: &Usage{
				InputTokens:                20,
				OutputTokens:               8,
				CachedInputTokens:          30,
				CacheCreationInputTokens:   40,
				CacheReadInputTokens:       50,
				CacheCreation5mInputTokens: 60,
				CacheCreation1hInputTokens: 70,
			},
		},
	}}

	res, err := Run(context.Background(), p)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	want := Usage{
		InputTokens:                30,
		OutputTokens:               10,
		CachedInputTokens:          33,
		CacheCreationInputTokens:   44,
		CacheReadInputTokens:       55,
		CacheCreation5mInputTokens: 66,
		CacheCreation1hInputTokens: 77,
	}
	if !reflect.DeepEqual(res.TotalUsage, want) {
		t.Fatalf("TotalUsage = %+v, want %+v", res.TotalUsage, want)
	}
	wantCalls := []Usage{
		{
			InputTokens:                10,
			OutputTokens:               2,
			CachedInputTokens:          3,
			CacheCreationInputTokens:   4,
			CacheReadInputTokens:       5,
			CacheCreation5mInputTokens: 6,
			CacheCreation1hInputTokens: 7,
		},
		{
			InputTokens:                20,
			OutputTokens:               8,
			CachedInputTokens:          30,
			CacheCreationInputTokens:   40,
			CacheReadInputTokens:       50,
			CacheCreation5mInputTokens: 60,
			CacheCreation1hInputTokens: 70,
		},
	}
	if !reflect.DeepEqual(res.CallUsage, wantCalls) {
		t.Fatalf("CallUsage = %+v, want %+v", res.CallUsage, wantCalls)
	}
	if len(res.CallUsage) != res.Steps {
		t.Fatalf("len(CallUsage) = %d, Steps = %d", len(res.CallUsage), res.Steps)
	}
}

func TestRunPauseViaBeforeTool(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "1", Name: "write"},
			{ID: "2", Name: "write"},
		}},
	}}}

	tool := Tool{ToolDef: ToolDef{Name: "write"}, Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
		return &ToolResult{ToolCallID: call.ID, Content: "ok"}, nil
	}}

	res, err := Run(context.Background(), p,
		WithTools(tool),
		WithBeforeTool(func(_ context.Context, _ ToolCall) (ToolAction, error) {
			return ToolActionPause, nil
		}),
	)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if res.StopReason != StopPaused {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopPaused)
	}
	if len(res.PendingToolCalls) != 2 {
		t.Fatalf("len(PendingToolCalls) = %d, want 2", len(res.PendingToolCalls))
	}
	if len(res.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1 (assistant only)", len(res.Messages))
	}
}

func TestRunSkipViaBeforeTool(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}}},
		{Message: Message{Role: RoleAssistant, Content: "next"}},
	}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
		return &ToolResult{ToolCallID: call.ID, Content: "ok"}, nil
	}}

	res, err := Run(context.Background(), p,
		WithTools(tool),
		WithBeforeTool(func(_ context.Context, _ ToolCall) (ToolAction, error) {
			return ToolActionSkip, nil
		}),
	)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if res.StopReason != StopEndTurn {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopEndTurn)
	}
	if len(res.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3", len(res.Messages))
	}
	if !res.Messages[1].ToolResult.IsError || res.Messages[1].ToolResult.Content != "Tool call skipped." {
		t.Fatalf("tool result = %+v", res.Messages[1].ToolResult)
	}
}

func TestRunUnknownToolReturnsToolErrorMessage(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "missing"}}}},
		{Message: Message{Role: RoleAssistant, Content: "done"}},
	}}

	res, err := Run(context.Background(), p)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if got := res.Messages[1].ToolResult.Content; got != "Unknown tool: missing" {
		t.Fatalf("unknown tool message = %q", got)
	}
	if !res.Messages[1].ToolResult.IsError {
		t.Fatalf("expected IsError=true")
	}
}

func TestRunStopsBeforeExecutingToolCallsAtMaxSteps(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{
			Message:      Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
			ResponseID:   "response-1",
			FinishReason: FinishReasonToolUse,
			Usage:        &Usage{InputTokens: 2, OutputTokens: 3},
		},
		{
			Message: Message{
				Role:      RoleAssistant,
				Content:   "I need one more tool result.",
				ToolCalls: []ToolCall{{ID: "2", Name: "echo"}},
			},
			ResponseID:    "response-2",
			FinishReason:  FinishReasonToolUse,
			FinishDetails: "tool requested",
			Usage:         &Usage{InputTokens: 5, OutputTokens: 7},
		},
	}}

	toolExecutions := 0
	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
		toolExecutions++
		return &ToolResult{ToolCallID: call.ID, Content: "ok"}, nil
	}}

	res, err := Run(context.Background(), p, WithTools(tool), WithMaxSteps(2))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopMaxSteps {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopMaxSteps)
	}
	if len(p.calls) != 2 {
		t.Fatalf("provider calls = %d, want 2", len(p.calls))
	}
	if toolExecutions != 1 {
		t.Fatalf("tool executions = %d, want 1", toolExecutions)
	}
	if res.Steps != 2 {
		t.Fatalf("Steps = %d, want 2", res.Steps)
	}
	if res.ResponseID != "response-2" {
		t.Fatalf("ResponseID = %q, want response-2", res.ResponseID)
	}
	if res.FinishReason != FinishReasonToolUse || res.FinishDetails != "tool requested" {
		t.Fatalf("finish = %q (%q), want %q (%q)", res.FinishReason, res.FinishDetails, FinishReasonToolUse, "tool requested")
	}
	if got := res.TotalUsage; got.InputTokens != 7 || got.OutputTokens != 10 {
		t.Fatalf("TotalUsage = %+v, want input=7 output=10", got)
	}
	if len(res.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3", len(res.Messages))
	}
	last := res.Messages[2]
	if last.Content != "I need one more tool result." || len(last.ToolCalls) != 1 || last.ToolCalls[0].ID != "2" {
		t.Fatalf("last assistant message = %+v", last)
	}
}

func TestRunEndTurnOnLastAllowedStepSucceeds(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message:      Message{Role: RoleAssistant, Content: "done"},
		FinishReason: FinishReasonEndTurn,
	}}}

	res, err := Run(context.Background(), p, WithMaxSteps(1))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopEndTurn {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopEndTurn)
	}
	if len(p.calls) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(p.calls))
	}
}

func TestRunProviderContinuationRespectsMaxSteps(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message:      Message{Role: RoleAssistant, Content: "continue me"},
		ResponseID:   "response-pause",
		FinishReason: FinishReasonContinuation,
	}}}

	res, err := Run(context.Background(), p, WithMaxSteps(1))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopMaxSteps {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopMaxSteps)
	}
	if res.FinishReason != FinishReasonContinuation {
		t.Fatalf("FinishReason = %q, want %q", res.FinishReason, FinishReasonContinuation)
	}
	if res.Steps != 1 || len(p.calls) != 1 {
		t.Fatalf("Steps = %d, provider calls = %d, want 1", res.Steps, len(p.calls))
	}
}

func TestRunCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res, err := Run(ctx, &mockProvider{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopCancelled {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopCancelled)
	}
}

func TestRunCancellationDuringProviderCall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &mockProvider{chat: func(ctx context.Context, _ ChatParams) (*ChatResult, error) {
		cancel()
		<-ctx.Done()
		return nil, ctx.Err()
	}}

	res, err := Run(ctx, p)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopCancelled {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopCancelled)
	}
	if len(p.calls) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(p.calls))
	}
}

func TestRunCancellationDuringToolBalancesToolCalls(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "1", Name: "write"},
			{ID: "2", Name: "write"},
		}},
	}}}
	tool := Tool{ToolDef: ToolDef{Name: "write"}, Execute: func(ctx context.Context, _ ToolCall) (*ToolResult, error) {
		cancel()
		return nil, ctx.Err()
	}}

	res, err := Run(ctx, p, WithTools(tool))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopCancelled {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopCancelled)
	}
	if len(res.Messages) != 3 {
		t.Fatalf("Messages = %+v, want assistant plus two tool results", res.Messages)
	}
	for i, msg := range res.Messages[1:] {
		if msg.ToolResult == nil || !msg.ToolResult.IsError || msg.ToolResult.ToolCallID != fmt.Sprint(i+1) {
			t.Fatalf("cancelled result %d = %+v", i, msg)
		}
	}
}

func TestRunRejectsInconsistentFinishReason(t *testing.T) {
	tests := []struct {
		name   string
		result *ChatResult
	}{
		{
			name: "end turn with tool calls",
			result: &ChatResult{
				Message:      Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
				FinishReason: FinishReasonEndTurn,
			},
		},
		{
			name: "tool use without tool calls",
			result: &ChatResult{
				Message:      Message{Role: RoleAssistant, Content: "missing call"},
				FinishReason: FinishReasonToolUse,
			},
		},
		{
			name: "continuation with tool calls",
			result: &ChatResult{
				Message:      Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
				FinishReason: FinishReasonContinuation,
			},
		},
		{
			name: "unknown reason",
			result: &ChatResult{
				Message:      Message{Role: RoleAssistant, Content: "done"},
				FinishReason: FinishReason("unexpected"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Run(context.Background(), &mockProvider{results: []*ChatResult{tt.result}})
			if err == nil {
				t.Fatal("Run() error = nil, want non-nil")
			}
			if !strings.Contains(err.Error(), "step 0") {
				t.Fatalf("Run() error = %q, want step prefix", err.Error())
			}
			if len(result.Messages) != 0 {
				t.Fatalf("Run() retained invalid provider message: %+v", result.Messages)
			}
		})
	}
}

func TestRunToolFilterControlsExposedTools(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{Message: Message{Role: RoleAssistant, Content: "done"}}}}

	allTools := []Tool{
		{ToolDef: ToolDef{Name: "zeta"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) { return &ToolResult{}, nil }},
		{ToolDef: ToolDef{Name: "alpha"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) { return &ToolResult{}, nil }},
	}

	filter := func(step int, _ []Message) []Tool {
		if step == 0 {
			return []Tool{allTools[0]}
		}
		return allTools
	}

	_, err := Run(context.Background(), p, WithTools(allTools...), WithToolFilter(filter))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(p.calls) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(p.calls))
	}
	gotNames := []string{}
	for _, def := range p.calls[0].Tools {
		gotNames = append(gotNames, def.Name)
	}
	if !reflect.DeepEqual(gotNames, []string{"zeta"}) {
		t.Fatalf("tool names = %v", gotNames)
	}
}

func TestRunPropagatesAfterToolError(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}, {ID: "2", Name: "echo"}}},
	}}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return &ToolResult{Content: "ok"}, nil
	}}

	wantErr := errors.New("after tool failed")
	res, err := Run(context.Background(), p, WithTools(tool), WithAfterTool(func(_ context.Context, _ ToolCall, _ *ToolResult) error {
		return wantErr
	}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run() error = %v, want %v", err, wantErr)
	}
	if res.StopReason != StopError {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopError)
	}
	if len(res.Messages) != 3 || res.Messages[1].ToolResult.Content != "ok" || !res.Messages[2].ToolResult.IsError {
		t.Fatalf("partial result = %+v, want completed call and unexecuted call result", res)
	}
}

func TestRunEventToolCallPointersAreStable(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}, {ID: "2", Name: "echo"}}}},
		{Message: Message{Role: RoleAssistant, Content: "done"}},
	}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
		return &ToolResult{ToolCallID: call.ID, Content: "ok"}, nil
	}}

	var seen []*ToolCall
	_, err := Run(context.Background(), p,
		WithTools(tool),
		WithEventHandler(func(e Event) {
			if e.Type == EventToolStart {
				seen = append(seen, e.ToolCall)
			}
		}),
	)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(seen) != 2 {
		t.Fatalf("tool start events = %d, want 2", len(seen))
	}
	if seen[0].ID != "1" || seen[1].ID != "2" {
		t.Fatalf("tool call IDs mutated: got %q and %q", seen[0].ID, seen[1].ID)
	}
}

func TestRunToolFilterInvalidToolReturnsError(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
	}}}

	valid := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
		return &ToolResult{ToolCallID: call.ID, Content: "ok"}, nil
	}}

	filter := func(_ int, _ []Message) []Tool {
		return []Tool{{ToolDef: ToolDef{Name: "echo"}}}
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Run() panicked with invalid filtered tool: %v", r)
		}
	}()

	_, err := Run(context.Background(), p, WithTools(valid), WithToolFilter(filter))
	if err == nil {
		t.Fatalf("Run() error = nil, want non-nil")
	}
}

func TestRunProviderErrorIsWrappedWithStep(t *testing.T) {
	wantErr := errors.New("provider failed")
	_, err := Run(context.Background(), &mockProvider{err: wantErr})
	if err == nil {
		t.Fatalf("Run() error = nil, want non-nil")
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run() error = %v, want wrapped %v", err, wantErr)
	}
	if !strings.Contains(err.Error(), "step 0") {
		t.Fatalf("Run() error = %q, want step prefix", err.Error())
	}
}

func TestRunProviderNilResultReturnsError(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{nil}}
	_, err := Run(context.Background(), p)
	if err == nil {
		t.Fatalf("Run() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "provider returned nil result") {
		t.Fatalf("Run() error = %q", err.Error())
	}
}

func TestRunBeforeToolErrorIsReturned(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}, {ID: "2", Name: "echo"}}},
	}}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return &ToolResult{Content: "ok"}, nil
	}}
	wantErr := errors.New("approval backend unavailable")

	res, err := Run(context.Background(), p,
		WithTools(tool),
		WithBeforeTool(func(_ context.Context, _ ToolCall) (ToolAction, error) {
			return ToolActionContinue, wantErr
		}),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run() error = %v, want %v", err, wantErr)
	}
	if res.StopReason != StopError {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopError)
	}
	if len(res.Messages) != 3 || !res.Messages[1].ToolResult.IsError || !res.Messages[2].ToolResult.IsError {
		t.Fatalf("partial result = %+v, want balanced unexecuted results", res)
	}
}

func TestRunRejectsInvalidBeforeToolAction(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
	}}}
	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		t.Fatal("tool executed for invalid action")
		return nil, nil
	}}

	res, err := Run(context.Background(), p, WithTools(tool), WithBeforeTool(func(_ context.Context, _ ToolCall) (ToolAction, error) {
		return ToolAction(99), nil
	}))
	if err == nil || !strings.Contains(err.Error(), "invalid action 99") {
		t.Fatalf("Run() error = %v, want invalid action", err)
	}
	if len(res.Messages) != 2 || res.Messages[1].ToolResult == nil || !res.Messages[1].ToolResult.IsError {
		t.Fatalf("result = %+v, want balanced tool error", res)
	}
}

func TestRunNilToolResultCreatesSanitisedFailure(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}}},
		{Message: Message{Role: RoleAssistant, Content: "done"}},
	}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return nil, nil
	}}

	res, err := Run(context.Background(), p, WithTools(tool))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(res.Messages) < 2 || res.Messages[1].ToolResult == nil {
		t.Fatalf("missing tool result message: %+v", res.Messages)
	}
	if !res.Messages[1].ToolResult.IsError {
		t.Fatalf("expected tool error result")
	}
	if res.Messages[1].ToolResult.Content != "Tool execution failed." {
		t.Fatalf("unexpected tool error content: %q", res.Messages[1].ToolResult.Content)
	}
}
