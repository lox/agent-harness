package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	harness "github.com/lox/agent-harness"
)

func TestProviderChatRequestAndDetailedResult(t *testing.T) {
	var request map[string]any
	server := newAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		request = decodeRequest(t, r)
		writeTextStream(w, "msg_fable", "claude-fable-5", "complete", "end_turn", "null", detailedUsageJSON)
	})
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
	result, err := p.Chat(context.Background(), harness.ChatParams{
		System:   "system prompt",
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "analyze this"}},
		Tools: []harness.ToolDef{{
			Name:        "read_file",
			Description: "Read a file.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
		}},
		Options: map[string]any{
			"max_tokens":      32768,
			"cache_ttl":       "1h",
			"thinking_budget": 4096,
		},
		Reasoning: harness.ReasoningOptions{Effort: "xhigh"},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if got := request["model"]; got != "claude-fable-5" {
		t.Fatalf("model = %v, want claude-fable-5", got)
	}
	if got := request["max_tokens"]; got != float64(32768) {
		t.Fatalf("max_tokens = %v, want 32768", got)
	}
	if got := nestedString(t, request, "output_config", "effort"); got != "xhigh" {
		t.Fatalf("output effort = %q, want xhigh", got)
	}
	if _, ok := request["thinking"]; ok {
		t.Fatalf("Fable request unexpectedly contains thinking: %v", request["thinking"])
	}
	if got := nestedString(t, request, "cache_control", "type"); got != "ephemeral" {
		t.Fatalf("cache control type = %q, want ephemeral", got)
	}
	if got := nestedString(t, request, "cache_control", "ttl"); got != "1h" {
		t.Fatalf("cache ttl = %q, want 1h", got)
	}
	if got := nestedString(t, firstObject(t, request, "system"), "text"); got != "system prompt" {
		t.Fatalf("system text = %q", got)
	}
	tool := firstObject(t, request, "tools")
	if tool["name"] != "read_file" || nestedString(t, tool, "input_schema", "type") != "object" {
		t.Fatalf("tool request = %#v", tool)
	}

	if result.ResponseID != "msg_fable" || result.FinishReason != harness.FinishReasonEndTurn {
		t.Fatalf("response identity/finish = %q/%q", result.ResponseID, result.FinishReason)
	}
	if result.Message.Content != "complete" || len(result.Message.ProviderData) == 0 {
		t.Fatalf("message = %+v", result.Message)
	}
	wantUsage := &harness.Usage{
		InputTokens:                11,
		OutputTokens:               7,
		CachedInputTokens:          40,
		CacheCreationInputTokens:   30,
		CacheReadInputTokens:       40,
		CacheCreation5mInputTokens: 20,
		CacheCreation1hInputTokens: 10,
	}
	if !reflect.DeepEqual(result.Usage, wantUsage) {
		t.Fatalf("usage = %+v, want %+v", result.Usage, wantUsage)
	}
}

func TestProviderAdaptiveThinkingByModel(t *testing.T) {
	tests := []struct {
		model        string
		wantThinking bool
	}{
		{model: "claude-fable-5"},
		{model: "claude-opus-4-8", wantThinking: true},
		{model: "claude-sonnet-5", wantThinking: true},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			var request map[string]any
			server := newAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
				request = decodeRequest(t, r)
				writeTextStream(w, "msg_model", tc.model, "done", "end_turn", "null", basicUsageJSON)
			})
			defer server.Close()

			p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
			_, err := p.Chat(context.Background(), harness.ChatParams{
				Model:    tc.model,
				Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
				Options:  map[string]any{"reasoning_effort": "low", "output_effort": "high"},
			})
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}

			thinking, hasThinking := request["thinking"].(map[string]any)
			if hasThinking != tc.wantThinking {
				t.Fatalf("thinking present = %v, want %v; request = %#v", hasThinking, tc.wantThinking, request)
			}
			if tc.wantThinking && thinking["type"] != "adaptive" {
				t.Fatalf("thinking = %#v, want adaptive", thinking)
			}
			if got := nestedString(t, request, "output_config", "effort"); got != "high" {
				t.Fatalf("output effort = %q, want high", got)
			}
		})
	}
}

func TestProviderRoundTripsToolAndThinkingBlocks(t *testing.T) {
	var request map[string]any
	server := newAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		request = decodeRequest(t, r)
		writeTextStream(w, "msg_done", "claude-opus-4-8", "done", "end_turn", "null", basicUsageJSON)
	})
	defer server.Close()

	providerData := json.RawMessage(`{"role":"assistant","content":[{"type":"thinking","thinking":"inspect first","signature":"sig_1"},{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"main.go"}}]}`)
	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
	_, err := p.Chat(context.Background(), harness.ChatParams{
		Model:  "claude-opus-4-8",
		System: "system",
		Messages: []harness.Message{
			{Role: harness.RoleUser, Content: "inspect"},
			{
				Role:         harness.RoleAssistant,
				Thinking:     "inspect first",
				ToolCalls:    []harness.ToolCall{{ID: "toolu_1", Name: "read_file", Arguments: json.RawMessage(`{"path":"main.go"}`)}},
				ProviderData: providerData,
			},
			{Role: harness.RoleTool, ToolResult: &harness.ToolResult{ToolCallID: "toolu_1", Content: "package main", IsError: false}},
		},
		Tools: []harness.ToolDef{{Name: "read_file", Parameters: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	messages := objectSlice(t, request, "messages")
	if len(messages) != 3 {
		t.Fatalf("messages = %#v", messages)
	}
	assistantBlocks := objectSlice(t, messages[1], "content")
	if assistantBlocks[0]["type"] != "thinking" || assistantBlocks[0]["signature"] != "sig_1" {
		t.Fatalf("thinking block = %#v", assistantBlocks[0])
	}
	if assistantBlocks[1]["type"] != "tool_use" || assistantBlocks[1]["id"] != "toolu_1" {
		t.Fatalf("tool block = %#v", assistantBlocks[1])
	}
	toolResult := firstObject(t, messages[2], "content")
	toolResultText := firstObject(t, toolResult, "content")
	if toolResult["type"] != "tool_result" || toolResult["tool_use_id"] != "toolu_1" || toolResultText["text"] != "package main" {
		t.Fatalf("tool result = %#v", toolResult)
	}
	if nestedString(t, request, "cache_control", "type") != "ephemeral" {
		t.Fatalf("growing conversation request lacks ephemeral cache control: %#v", request)
	}
}

func TestProviderStreamingAccumulationAndDeltas(t *testing.T) {
	server := newAnthropicServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse("message_start", `{"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","model":"claude-opus-4-8","content":[],"stop_reason":null,"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":5,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`))
		_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`))
		_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"consider"}}`))
		_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_stream"}}`))
		_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
		_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`))
		_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Inspecting"}}`))
		_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":1}`))
		_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_stream","name":"read_file","input":{}}}`))
		_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"main.go\"}"}}`))
		_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":2}`))
		_, _ = io.WriteString(w, sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null,"stop_details":null},"usage":{"input_tokens":5,"output_tokens":9,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`))
		_, _ = io.WriteString(w, sse("message_stop", `{"type":"message_stop"}`))
	})
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
	var deltas []harness.Delta
	result, err := p.Chat(context.Background(), harness.ChatParams{
		Model:    "claude-opus-4-8",
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "inspect"}},
		OnDelta:  func(delta harness.Delta) { deltas = append(deltas, delta) },
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.ResponseID != "msg_stream" || result.FinishReason != harness.FinishReasonToolUse {
		t.Fatalf("result identity/finish = %q/%q", result.ResponseID, result.FinishReason)
	}
	if result.Message.Thinking != "consider" || result.Message.Content != "Inspecting" {
		t.Fatalf("message text/thinking = %q/%q", result.Message.Content, result.Message.Thinking)
	}
	if len(result.Message.ToolCalls) != 1 || result.Message.ToolCalls[0].ID != "toolu_stream" || string(result.Message.ToolCalls[0].Arguments) != `{"path":"main.go"}` {
		t.Fatalf("tool calls = %+v", result.Message.ToolCalls)
	}
	if len(result.Message.ProviderData) == 0 || !strings.Contains(string(result.Message.ProviderData), `"signature":"sig_stream"`) {
		t.Fatalf("provider data did not retain signature: %s", result.Message.ProviderData)
	}

	var textDelta, thinkingDelta string
	var toolDeltas []harness.ToolCallDelta
	for _, delta := range deltas {
		textDelta += delta.Text
		thinkingDelta += delta.Thinking
		if delta.ToolCall != nil {
			toolDeltas = append(toolDeltas, *delta.ToolCall)
		}
	}
	if textDelta != "Inspecting" || thinkingDelta != "consider" {
		t.Fatalf("stream deltas text/thinking = %q/%q", textDelta, thinkingDelta)
	}
	if len(toolDeltas) != 2 || toolDeltas[0].Index != 0 || toolDeltas[1].Index != 0 || toolDeltas[0].ID != "toolu_stream" || toolDeltas[0].Name != "read_file" || toolDeltas[1].Arguments != `{"path":"main.go"}` {
		t.Fatalf("tool deltas = %+v", toolDeltas)
	}
}

func TestProviderJoinsTextBlocksWithNewlines(t *testing.T) {
	server := newAnthropicServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse("message_start", `{"type":"message_start","message":{"id":"msg_text_blocks","type":"message","role":"assistant","model":"claude-fable-5","content":[],"stop_reason":null,"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":2,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`))
		_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
		_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first block"}}`))
		_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
		_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`))
		_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"second block"}}`))
		_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":1}`))
		_, _ = io.WriteString(w, sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"stop_details":null},"usage":{"input_tokens":2,"output_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`))
		_, _ = io.WriteString(w, sse("message_stop", `{"type":"message_stop"}`))
	})
	defer server.Close()

	result, err := New(WithBaseURL(server.URL), WithAPIKey("test-key")).Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.Message.Content != "first block\nsecond block" {
		t.Fatalf("message content = %q, want newline-separated blocks", result.Message.Content)
	}
}

func TestProviderStreamingAndNonStreamingCallersAreEquivalent(t *testing.T) {
	server := newAnthropicServer(t, func(w http.ResponseWriter, _ *http.Request) {
		writeTextStream(w, "msg_same", "claude-fable-5", "same", "end_turn", "null", basicUsageJSON)
	})
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
	params := harness.ChatParams{Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}}}
	withoutDeltas, err := p.Chat(context.Background(), params)
	if err != nil {
		t.Fatalf("Chat() without callback error = %v", err)
	}
	params.OnDelta = func(harness.Delta) {}
	withDeltas, err := p.Chat(context.Background(), params)
	if err != nil {
		t.Fatalf("Chat() with callback error = %v", err)
	}
	if !reflect.DeepEqual(withoutDeltas, withDeltas) {
		t.Fatalf("results differ:\nwithout = %+v\nwith = %+v", withoutDeltas, withDeltas)
	}
}

func TestProviderStopReasons(t *testing.T) {
	tests := []struct {
		name        string
		stopReason  string
		stopDetails string
		withTool    bool
		wantReason  harness.FinishReason
		wantDetails string
	}{
		{name: "end turn", stopReason: "end_turn", stopDetails: "null", wantReason: harness.FinishReasonEndTurn},
		{name: "stop sequence", stopReason: "stop_sequence", stopDetails: "null", wantReason: harness.FinishReasonEndTurn},
		{name: "tool use", stopReason: "tool_use", stopDetails: "null", withTool: true, wantReason: harness.FinishReasonToolUse},
		{name: "max tokens", stopReason: "max_tokens", stopDetails: "null", wantReason: harness.FinishReasonMaxTokens},
		{name: "pause turn", stopReason: "pause_turn", stopDetails: "null", wantReason: harness.FinishReasonContinuation},
		{name: "refusal", stopReason: "refusal", stopDetails: `{"type":"refusal","category":"cyber","explanation":"declined"}`, wantReason: harness.FinishReasonRefusal, wantDetails: "cyber: declined"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := newAnthropicServer(t, func(w http.ResponseWriter, _ *http.Request) {
				if tc.withTool {
					writeToolStream(w, "msg_stop", "claude-fable-5", tc.stopReason, tc.stopDetails)
					return
				}
				writeTextStream(w, "msg_stop", "claude-fable-5", "partial", tc.stopReason, tc.stopDetails, basicUsageJSON)
			})
			defer server.Close()

			p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
			result, err := p.Chat(context.Background(), harness.ChatParams{Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}}})
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}
			if result.FinishReason != tc.wantReason || result.FinishDetails != tc.wantDetails {
				t.Fatalf("finish = %q (%q), want %q (%q)", result.FinishReason, result.FinishDetails, tc.wantReason, tc.wantDetails)
			}
			if result.ResponseID != "msg_stop" {
				t.Fatalf("response ID = %q", result.ResponseID)
			}
		})
	}
}

func TestProviderPauseTurnContinuesThroughRun(t *testing.T) {
	requestCount := 0
	var continuationRequest map[string]any
	server := newAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		request := decodeRequest(t, r)
		if got := nestedString(t, request, "cache_control", "type"); got != "ephemeral" {
			t.Fatalf("request %d cache control type = %q", requestCount, got)
		}
		if got := nestedString(t, request, "cache_control", "ttl"); got != "1h" {
			t.Fatalf("request %d cache ttl = %q", requestCount, got)
		}
		if requestCount == 1 {
			writeTextStream(w, "msg_pause", "claude-fable-5", "working", "pause_turn", "null", firstDetailedUsageJSON)
			return
		}
		continuationRequest = request
		writeTextStream(w, "msg_final", "claude-fable-5", "complete", "end_turn", "null", secondDetailedUsageJSON)
	})
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
	result, err := harness.Run(context.Background(), p,
		harness.WithMessages(harness.Message{Role: harness.RoleUser, Content: "analyze"}),
		harness.WithProviderOptions(map[string]any{"cache_ttl": "1h"}),
	)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if requestCount != 2 || result.Steps != 2 {
		t.Fatalf("requests/steps = %d/%d, want 2/2", requestCount, result.Steps)
	}
	if result.StopReason != harness.StopEndTurn || result.FinishReason != harness.FinishReasonEndTurn || result.ResponseID != "msg_final" {
		t.Fatalf("terminal result = stop %v, finish %q, ID %q", result.StopReason, result.FinishReason, result.ResponseID)
	}
	wantCalls := []harness.Usage{
		{InputTokens: 2, OutputTokens: 1, CachedInputTokens: 4, CacheCreationInputTokens: 3, CacheReadInputTokens: 4, CacheCreation5mInputTokens: 2, CacheCreation1hInputTokens: 1},
		{InputTokens: 5, OutputTokens: 6, CachedInputTokens: 8, CacheCreationInputTokens: 7, CacheReadInputTokens: 8, CacheCreation5mInputTokens: 3, CacheCreation1hInputTokens: 4},
	}
	if !reflect.DeepEqual(result.CallUsage, wantCalls) {
		t.Fatalf("call usage = %+v, want %+v", result.CallUsage, wantCalls)
	}
	wantTotal := harness.Usage{InputTokens: 7, OutputTokens: 7, CachedInputTokens: 12, CacheCreationInputTokens: 10, CacheReadInputTokens: 12, CacheCreation5mInputTokens: 5, CacheCreation1hInputTokens: 5}
	if !reflect.DeepEqual(result.TotalUsage, wantTotal) {
		t.Fatalf("total usage = %+v, want %+v", result.TotalUsage, wantTotal)
	}

	messages := objectSlice(t, continuationRequest, "messages")
	if len(messages) != 2 {
		t.Fatalf("continuation messages = %#v", messages)
	}
	continuedBlock := firstObject(t, messages[1], "content")
	if messages[1]["role"] != "assistant" || continuedBlock["type"] != "text" || continuedBlock["text"] != "working" {
		t.Fatalf("continued assistant message = %#v", messages[1])
	}
}

func TestProviderTerminalStopsThroughRun(t *testing.T) {
	tests := []struct {
		name        string
		stopReason  string
		stopDetails string
		wantStop    harness.StopReason
		wantFinish  harness.FinishReason
		wantDetails string
	}{
		{name: "max tokens", stopReason: "max_tokens", stopDetails: "null", wantStop: harness.StopIncomplete, wantFinish: harness.FinishReasonMaxTokens},
		{name: "refusal", stopReason: "refusal", stopDetails: `{"type":"refusal","category":"cyber","explanation":"declined"}`, wantStop: harness.StopRefusal, wantFinish: harness.FinishReasonRefusal, wantDetails: "cyber: declined"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := newAnthropicServer(t, func(w http.ResponseWriter, _ *http.Request) {
				writeTextStream(w, "msg_terminal", "claude-fable-5", "partial", tc.stopReason, tc.stopDetails, basicUsageJSON)
			})
			defer server.Close()

			p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
			result, err := harness.Run(context.Background(), p, harness.WithMessages(harness.Message{Role: harness.RoleUser, Content: "analyze"}))
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if result.StopReason != tc.wantStop || result.FinishReason != tc.wantFinish || result.FinishDetails != tc.wantDetails {
				t.Fatalf("terminal result = stop %v, finish %q (%q)", result.StopReason, result.FinishReason, result.FinishDetails)
			}
		})
	}
}

func TestProviderPromptCacheCanBeDisabled(t *testing.T) {
	var request map[string]any
	server := newAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		request = decodeRequest(t, r)
		writeTextStream(w, "msg_no_cache", "claude-fable-5", "done", "end_turn", "null", basicUsageJSON)
	})
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"))
	_, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		Options:  map[string]any{"prompt_cache": false},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if _, ok := request["cache_control"]; ok {
		t.Fatalf("request contains cache_control: %#v", request["cache_control"])
	}
}

func newAnthropicServer(t *testing.T, handler func(http.ResponseWriter, *http.Request)) *httptest.Server {
	t.Helper()
	var mu sync.Mutex
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %q, want /v1/messages", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		handler(w, r)
	}))
}

func decodeRequest(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var request map[string]any
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return request
}

func nestedString(t *testing.T, object map[string]any, path ...string) string {
	t.Helper()
	current := object
	for _, key := range path[:len(path)-1] {
		next, ok := current[key].(map[string]any)
		if !ok {
			t.Fatalf("%s is %T, want object in %#v", key, current[key], object)
		}
		current = next
	}
	value, ok := current[path[len(path)-1]].(string)
	if !ok {
		t.Fatalf("%s is %T, want string in %#v", path[len(path)-1], current[path[len(path)-1]], object)
	}
	return value
}

func objectSlice(t *testing.T, object map[string]any, key string) []map[string]any {
	t.Helper()
	raw, ok := object[key].([]any)
	if !ok {
		t.Fatalf("%s is %T, want array in %#v", key, object[key], object)
	}
	objects := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			t.Fatalf("%s item is %T, want object", key, item)
		}
		objects = append(objects, entry)
	}
	return objects
}

func firstObject(t *testing.T, object map[string]any, key string) map[string]any {
	t.Helper()
	objects := objectSlice(t, object, key)
	if len(objects) == 0 {
		t.Fatalf("%s is empty in %#v", key, object)
	}
	return objects[0]
}

func writeTextStream(w http.ResponseWriter, id, model, text, stopReason, stopDetails, usage string) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, sse("message_start", fmt.Sprintf(`{"type":"message_start","message":{"id":%q,"type":"message","role":"assistant","model":%q,"content":[],"stop_reason":null,"stop_sequence":null,"stop_details":null,"usage":%s}}`, id, model, usage)))
	_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
	_, _ = io.WriteString(w, sse("content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text)))
	_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
	_, _ = io.WriteString(w, sse("message_delta", fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%q,"stop_sequence":null,"stop_details":%s},"usage":%s}`, stopReason, stopDetails, usage)))
	_, _ = io.WriteString(w, sse("message_stop", `{"type":"message_stop"}`))
}

func writeToolStream(w http.ResponseWriter, id, model, stopReason, stopDetails string) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, sse("message_start", fmt.Sprintf(`{"type":"message_start","message":{"id":%q,"type":"message","role":"assistant","model":%q,"content":[],"stop_reason":null,"stop_sequence":null,"stop_details":null,"usage":%s}}`, id, model, basicUsageJSON)))
	_, _ = io.WriteString(w, sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"echo","input":{}}}`))
	_, _ = io.WriteString(w, sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"text\":\"hi\"}"}}`))
	_, _ = io.WriteString(w, sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
	_, _ = io.WriteString(w, sse("message_delta", fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%q,"stop_sequence":null,"stop_details":%s},"usage":%s}`, stopReason, stopDetails, basicUsageJSON)))
	_, _ = io.WriteString(w, sse("message_stop", `{"type":"message_stop"}`))
}

func sse(event, data string) string {
	return "event: " + event + "\n" + "data: " + data + "\n\n"
}

const basicUsageJSON = `{"input_tokens":2,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}`

const detailedUsageJSON = `{"input_tokens":11,"output_tokens":7,"cache_creation_input_tokens":30,"cache_read_input_tokens":40,"cache_creation":{"ephemeral_5m_input_tokens":20,"ephemeral_1h_input_tokens":10}}`

const firstDetailedUsageJSON = `{"input_tokens":2,"output_tokens":1,"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":1}}`

const secondDetailedUsageJSON = `{"input_tokens":5,"output_tokens":6,"cache_creation_input_tokens":7,"cache_read_input_tokens":8,"cache_creation":{"ephemeral_5m_input_tokens":3,"ephemeral_1h_input_tokens":4}}`
