package harness

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type mockProvider struct {
	results []*ChatResult
	err     error
	calls   []ChatParams
}

func (m *mockProvider) Chat(_ context.Context, params ChatParams) (*ChatResult, error) {
	m.calls = append(m.calls, params)
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
		Message: Message{Role: RoleAssistant, Content: "done"},
		Usage:   &Usage{InputTokens: 3, OutputTokens: 5},
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
	if got := res.TotalUsage; got.InputTokens != 3 || got.OutputTokens != 5 {
		t.Fatalf("TotalUsage = %+v", got)
	}
	if len(res.Messages) != 1 || res.Messages[0].Content != "done" {
		t.Fatalf("Messages = %+v", res.Messages)
	}
}

func TestRunExecutesToolCalls(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{
		{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}}},
		{Message: Message{Role: RoleAssistant, Content: "final"}},
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

func TestRunStopsAtMaxSteps(t *testing.T) {
	p := &mockProvider{results: []*ChatResult{{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
	}}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, call ToolCall) (*ToolResult, error) {
		return &ToolResult{ToolCallID: call.ID, Content: "ok"}, nil
	}}

	res, err := Run(context.Background(), p, WithTools(tool), WithMaxSteps(1))
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if res.StopReason != StopMaxSteps {
		t.Fatalf("StopReason = %v, want %v", res.StopReason, StopMaxSteps)
	}
	if len(res.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(res.Messages))
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
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
	}}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return &ToolResult{Content: "ok"}, nil
	}}

	wantErr := errors.New("after tool failed")
	_, err := Run(context.Background(), p, WithTools(tool), WithAfterTool(func(_ context.Context, _ ToolCall, _ *ToolResult) error {
		return wantErr
	}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run() error = %v, want %v", err, wantErr)
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
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "1", Name: "echo"}}},
	}}}

	tool := Tool{ToolDef: ToolDef{Name: "echo"}, Execute: func(_ context.Context, _ ToolCall) (*ToolResult, error) {
		return &ToolResult{Content: "ok"}, nil
	}}
	wantErr := errors.New("approval backend unavailable")

	_, err := Run(context.Background(), p,
		WithTools(tool),
		WithBeforeTool(func(_ context.Context, _ ToolCall) (ToolAction, error) {
			return ToolActionContinue, wantErr
		}),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run() error = %v, want %v", err, wantErr)
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
