package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	harness "github.com/lox/agent-harness"
)

func TestProviderRequestMapsHistoryToolsAndOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("path = %q, want /responses", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		if got := body["model"]; got != "gpt-5.4" {
			t.Fatalf("model = %#v, want gpt-5.4", got)
		}
		if got := body["instructions"]; got != "base system\n\nhistory system" {
			t.Fatalf("instructions = %#v", got)
		}
		if got := body["prompt_cache_key"]; got != "analysis-session" {
			t.Fatalf("prompt_cache_key = %#v", got)
		}
		if got := body["max_output_tokens"]; got != float64(8192) {
			t.Fatalf("max_output_tokens = %#v", got)
		}
		text := body["text"].(map[string]any)
		format := text["format"].(map[string]any)
		if got := format["type"]; got != "json_object" {
			t.Fatalf("text.format.type = %#v, want json_object", got)
		}
		reasoning := body["reasoning"].(map[string]any)
		if got := reasoning["effort"]; got != "xhigh" {
			t.Fatalf("reasoning.effort = %#v", got)
		}

		input := body["input"].([]any)
		if len(input) != 5 {
			t.Fatalf("input length = %d, want 5: %#v", len(input), input)
		}
		wantRoles := map[int]string{0: "user", 1: "assistant", 4: "user"}
		for i, want := range wantRoles {
			if got := input[i].(map[string]any)["role"]; got != want {
				t.Fatalf("input[%d].role = %#v, want %q", i, got, want)
			}
		}
		for i, want := range map[int]string{2: "function_call", 3: "function_call_output"} {
			if got := input[i].(map[string]any)["type"]; got != want {
				t.Fatalf("input[%d].type = %#v, want %q", i, got, want)
			}
		}
		call := input[2].(map[string]any)
		if call["call_id"] != "call_1" || call["name"] != "lookup" {
			t.Fatalf("function call input = %#v", call)
		}
		output := input[3].(map[string]any)
		if output["call_id"] != "call_1" || output["output"] != "found" {
			t.Fatalf("function output input = %#v", output)
		}

		tools := body["tools"].([]any)
		if len(tools) != 1 {
			t.Fatalf("tools = %#v", tools)
		}
		tool := tools[0].(map[string]any)
		if tool["type"] != "function" || tool["name"] != "lookup" {
			t.Fatalf("tool = %#v", tool)
		}
		if tool["description"] != "look something up" || tool["strict"] != false {
			t.Fatalf("tool metadata = %#v", tool)
		}
		parameters := tool["parameters"].(map[string]any)
		if parameters["type"] != "object" || parameters["properties"] == nil {
			t.Fatalf("tool parameters = %#v", parameters)
		}
		required := parameters["required"].([]any)
		if len(required) != 1 || required[0] != "q" {
			t.Fatalf("tool required fields = %#v, want [q]", required)
		}
		if _, nested := tool["function"]; nested {
			t.Fatalf("Responses tool unexpectedly used Chat Completions wrapper: %#v", tool)
		}

		writeJSON(t, w, completedTextResponse("resp_request", "done", 17, 9, 11))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel("gpt-5.4"))
	result, err := p.Chat(context.Background(), harness.ChatParams{
		System: "base system",
		Messages: []harness.Message{
			{Role: harness.RoleSystem, Content: "history system"},
			{Role: harness.RoleUser, Content: "hello"},
			{Role: harness.RoleAssistant, Content: "checking", ToolCalls: []harness.ToolCall{{
				ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`),
			}}},
			{Role: harness.RoleTool, ToolResult: &harness.ToolResult{ToolCallID: "call_1", Content: "found"}},
			{Role: harness.RoleUser, Content: "continue"},
		},
		Tools: []harness.ToolDef{{
			Name:        "lookup",
			Description: "look something up",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"},"limit":{"type":"integer"}},"required":["q"]}`),
		}},
		Options: map[string]any{
			"prompt_cache_key":  "analysis-session",
			"reasoning_effort":  "xhigh",
			"max_output_tokens": 8192,
			"response_format":   "json_object",
		},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.ResponseID != "resp_request" || messageResponseIDForTest(t, result.Message) != "resp_request" {
		t.Fatalf("response IDs = result %q message %q", result.ResponseID, messageResponseIDForTest(t, result.Message))
	}
	if result.FinishReason != harness.FinishReasonEndTurn {
		t.Fatalf("finish reason = %q", result.FinishReason)
	}
	if result.Usage == nil || result.Usage.InputTokens != 8 || result.Usage.CachedInputTokens != 9 || result.Usage.CacheReadInputTokens != 9 || result.Usage.OutputTokens != 11 {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestRunContinuesWithPreviousResponseIDAndFunctionOutput(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := atomic.AddInt32(&calls, 1)
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if got := body["prompt_cache_key"]; got != "deep-analysis-run" {
			t.Fatalf("request %d prompt_cache_key = %#v", call, got)
		}
		reasoning := body["reasoning"].(map[string]any)
		if reasoning["mode"] != "pro" || reasoning["effort"] != "xhigh" {
			t.Fatalf("request %d reasoning = %#v", call, reasoning)
		}

		switch call {
		case 1:
			if got := body["instructions"]; got != "stay concise" {
				t.Fatalf("first instructions = %#v", got)
			}
			if got := body["previous_response_id"]; got != "resp_existing" {
				t.Fatalf("first previous_response_id = %#v, want resp_existing", got)
			}
			writeJSON(t, w, toolCallResponseWithUsage("resp_1", "call_1", "echo", `{"text":"hi"}`, 60, 10, 20, 5))
		case 2:
			if got := body["instructions"]; got != "stay concise" {
				t.Fatalf("continuation instructions = %#v", got)
			}
			if got := body["previous_response_id"]; got != "resp_1" {
				t.Fatalf("previous_response_id = %#v, want resp_1", got)
			}
			input := body["input"].([]any)
			if len(input) != 1 {
				t.Fatalf("continuation input = %#v, want only tool output", input)
			}
			output := input[0].(map[string]any)
			if output["type"] != "function_call_output" || output["call_id"] != "call_1" || output["output"] != "echoed: hi" {
				t.Fatalf("continuation output = %#v", output)
			}
			writeJSON(t, w, completedTextResponseWithUsage("resp_2", "finished", 120, 40, 30, 8))
		default:
			t.Fatalf("unexpected request %d", call)
		}
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel("gpt-5.4"))
	result, err := harness.Run(context.Background(), p,
		harness.WithSystem("stay concise"),
		harness.WithMessages(harness.Message{Role: harness.RoleUser, Content: "echo hi"}),
		harness.WithModel("gpt-5.6"),
		harness.WithPreviousResponseID("resp_existing"),
		harness.WithReasoning(harness.ReasoningOptions{Effort: "xhigh", Mode: "pro"}),
		harness.WithProviderOptions(map[string]any{"prompt_cache_key": "deep-analysis-run"}),
		harness.WithTools(harness.Tool{
			ToolDef: harness.ToolDef{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)},
			Execute: func(_ context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
				return &harness.ToolResult{ToolCallID: call.ID, Content: "echoed: hi"}, nil
			},
		}),
	)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if atomic.LoadInt32(&calls) != 2 || result.Steps != 2 {
		t.Fatalf("calls = %d, steps = %d", calls, result.Steps)
	}
	if result.ResponseID != "resp_2" || result.FinishReason != harness.FinishReasonEndTurn || result.StopReason != harness.StopEndTurn {
		t.Fatalf("terminal result = %+v", result)
	}
	if len(result.Messages) != 3 || messageResponseIDForTest(t, result.Messages[0]) != "resp_1" || messageResponseIDForTest(t, result.Messages[2]) != "resp_2" {
		t.Fatalf("messages = %+v", result.Messages)
	}
	wantCalls := []harness.Usage{
		{InputTokens: 30, OutputTokens: 5, CachedInputTokens: 10, CacheCreationInputTokens: 20, CacheReadInputTokens: 10},
		{InputTokens: 50, OutputTokens: 8, CachedInputTokens: 40, CacheCreationInputTokens: 30, CacheReadInputTokens: 40},
	}
	if !reflect.DeepEqual(result.CallUsage, wantCalls) {
		t.Fatalf("call usage = %+v, want %+v", result.CallUsage, wantCalls)
	}
	wantTotal := harness.Usage{InputTokens: 80, OutputTokens: 13, CachedInputTokens: 50, CacheCreationInputTokens: 50, CacheReadInputTokens: 50}
	if !reflect.DeepEqual(result.TotalUsage, wantTotal) {
		t.Fatalf("total usage = %+v, want %+v", result.TotalUsage, wantTotal)
	}
}

func TestExplicitPreviousResponseIDUsesMatchingHistorySuffix(t *testing.T) {
	p := New(WithDefaultModel("gpt-5.4"))
	request, err := p.buildRequest(harness.ChatParams{
		Messages: []harness.Message{
			{Role: harness.RoleSystem, Content: "history instructions"},
			{Role: harness.RoleUser, Content: "old input"},
			{Role: harness.RoleAssistant, ProviderData: providerDataForTest(t, "resp_old"), ToolCalls: []harness.ToolCall{{
				ID: "call_old", Name: "echo", Arguments: json.RawMessage(`{}`),
			}}},
			{Role: harness.RoleTool, ToolResult: &harness.ToolResult{ToolCallID: "call_old", Content: "done"}},
		},
		Options: map[string]any{"previous_response_id": "resp_old"},
	})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}
	bodyBytes, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if body["previous_response_id"] != "resp_old" || body["instructions"] != "history instructions" {
		t.Fatalf("continuation fields = %#v", body)
	}
	input := body["input"].([]any)
	if len(input) != 1 || input[0].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("continuation input = %#v", input)
	}
}

func TestBuildRequestOptsIntoStrictToolsWithoutRewritingSchema(t *testing.T) {
	p := New(WithDefaultModel("gpt-5.4"))
	request, err := p.buildRequest(harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		Tools: []harness.ToolDef{{
			Name:       "lookup",
			Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"},"limit":{"type":"integer"}},"required":["q"]}`),
		}},
		Options: map[string]any{"strict_tools": true},
	})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}

	tool := request.Tools[0].OfFunction
	if !tool.Strict.Valid() || !tool.Strict.Value {
		t.Fatalf("tool strict = %#v, want true", tool.Strict)
	}
	required := tool.Parameters["required"].([]any)
	if len(required) != 1 || required[0] != "q" {
		t.Fatalf("tool required fields = %#v, want unchanged [q]", required)
	}
}

func TestProviderNormalizesTerminalStates(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		finish  harness.FinishReason
		content string
	}{
		{name: "completed", body: completedTextResponse("resp_done", "done", 1, 0, 1), finish: harness.FinishReasonEndTurn, content: "done"},
		{name: "max output", body: incompleteResponse("resp_max", "max_output_tokens"), finish: harness.FinishReasonMaxTokens},
		{name: "other incomplete", body: incompleteResponse("resp_filter", "content_filter"), finish: harness.FinishReasonIncomplete},
		{name: "refusal", body: refusalResponse("resp_refusal", "I cannot help with that."), finish: harness.FinishReasonRefusal, content: "I cannot help with that."},
		{name: "tool calls", body: toolCallResponse("resp_tool", "call_9", "lookup", `{}`), finish: harness.FinishReasonToolUse},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, tt.body)
			}))
			defer server.Close()

			result, err := New(WithBaseURL(server.URL), WithAPIKey("test")).Chat(context.Background(), harness.ChatParams{
				Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
			})
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}
			if result.FinishReason != tt.finish || result.Message.Content != tt.content {
				t.Fatalf("result = %+v, want finish %q content %q", result, tt.finish, tt.content)
			}
		})
	}
}

func TestProviderJoinsOutputTextBlocksWithNewlines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, `{
			"id":"resp_text_blocks",
			"object":"response",
			"status":"completed",
			"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[
				{"type":"output_text","text":"first block","annotations":[]},
				{"type":"output_text","text":"second block","annotations":[]}
			]}],
			"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":4,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":5}
		}`)
	}))
	defer server.Close()

	result, err := New(WithBaseURL(server.URL), WithAPIKey("test")).Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.Message.Content != "first block\nsecond block" {
		t.Fatalf("message content = %q, want newline-separated blocks", result.Message.Content)
	}
}

func TestProviderNormalizesEmptyToolCallArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, toolCallResponse("resp_tool", "call_1", "time_now", ""))
	}))
	defer server.Close()

	result, err := New(WithBaseURL(server.URL), WithAPIKey("test")).Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "what time is it?"}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(result.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(result.Message.ToolCalls))
	}
	if got := string(result.Message.ToolCalls[0].Arguments); got != `{}` {
		t.Fatalf("tool arguments = %q, want {}", got)
	}
}

func TestProviderStreamingAssemblesEquivalentFinalResponse(t *testing.T) {
	terminal := combinedResponse("resp_stream", "call_stream", "lookup", `{"q":"docs"}`, "answer")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"stream":true`) {
			writeJSON(t, w, terminal)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, "response.output_item.added", `{"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_stream","name":"lookup","arguments":"","status":"in_progress"}}`)
		writeSSE(t, w, "response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","sequence_number":2,"delta":"{\"q\":"}`)
		writeSSE(t, w, "response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","sequence_number":3,"delta":"\"docs\"}"}`)
		writeSSE(t, w, "response.output_text.delta", `{"type":"response.output_text.delta","output_index":1,"content_index":0,"item_id":"msg_1","sequence_number":4,"delta":"ans","logprobs":[]}`)
		writeSSE(t, w, "response.output_text.delta", `{"type":"response.output_text.delta","output_index":1,"content_index":0,"item_id":"msg_1","sequence_number":5,"delta":"wer","logprobs":[]}`)
		writeSSE(t, w, "response.completed", fmt.Sprintf(`{"type":"response.completed","sequence_number":6,"response":%s}`, compactJSON(t, terminal)))
	}))
	defer server.Close()

	var textDeltas []string
	var toolDeltas []harness.ToolCallDelta
	p := New(WithBaseURL(server.URL), WithAPIKey("test"))
	params := harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta: func(delta harness.Delta) {
			if delta.Text != "" {
				textDeltas = append(textDeltas, delta.Text)
			}
			if delta.ToolCall != nil {
				toolDeltas = append(toolDeltas, *delta.ToolCall)
			}
		},
	}
	result, err := p.Chat(context.Background(), params)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if !reflect.DeepEqual(textDeltas, []string{"ans", "wer"}) {
		t.Fatalf("text deltas = %#v", textDeltas)
	}
	if len(toolDeltas) != 3 || toolDeltas[0].ID != "call_stream" || toolDeltas[0].Name != "lookup" || toolDeltas[1].Arguments+toolDeltas[2].Arguments != `{"q":"docs"}` {
		t.Fatalf("tool deltas = %+v", toolDeltas)
	}
	if result.ResponseID != "resp_stream" || result.Message.Content != "answer" || result.FinishReason != harness.FinishReasonToolUse {
		t.Fatalf("result = %+v", result)
	}
	if len(result.Message.ToolCalls) != 1 || result.Message.ToolCalls[0].ID != "call_stream" || string(result.Message.ToolCalls[0].Arguments) != `{"q":"docs"}` {
		t.Fatalf("final tool calls = %+v", result.Message.ToolCalls)
	}
	if result.Usage == nil || result.Usage.CachedInputTokens != 6 {
		t.Fatalf("usage = %+v", result.Usage)
	}

	params.OnDelta = nil
	nonStreaming, err := p.Chat(context.Background(), params)
	if err != nil {
		t.Fatalf("non-streaming Chat() error = %v", err)
	}
	if !reflect.DeepEqual(result, nonStreaming) {
		t.Fatalf("streaming and non-streaming results differ:\nstreaming: %+v\nnon-streaming: %+v", result, nonStreaming)
	}
}

func TestProviderStreamingReturnsErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, "error", `{"type":"error","sequence_number":1,"code":"server_error","message":"try again later","param":""}`)
	}))
	defer server.Close()

	_, err := New(WithBaseURL(server.URL), WithAPIKey("test")).Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta:  func(harness.Delta) {},
	})
	if err == nil || !strings.Contains(err.Error(), "try again later") {
		t.Fatalf("Chat() error = %v, want streamed API error", err)
	}
}

func TestProviderRejectsNonTerminalResponseStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, `{"id":"resp_queued","object":"response","status":"queued","output":[]}`)
	}))
	defer server.Close()

	_, err := New(WithBaseURL(server.URL), WithAPIKey("test")).Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
	})
	if err == nil || !strings.Contains(err.Error(), `unexpected status "queued"`) {
		t.Fatalf("Chat() error = %v, want unexpected-status error", err)
	}
}

func completedTextResponse(id, text string, input, cached, output int) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":%q,"annotations":[]}]}],
		"usage":{"input_tokens":%d,"input_tokens_details":{"cached_tokens":%d},"output_tokens":%d,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":%d}
	}`, id, text, input, cached, output, input+output)
}

func toolCallResponse(id, callID, name, arguments string) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"completed",
		"output":[{"type":"function_call","id":"fc_1","status":"completed","call_id":%q,"name":%q,"arguments":%q}],
		"usage":{"input_tokens":3,"input_tokens_details":{"cached_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":5}
	}`, id, callID, name, arguments)
}

func toolCallResponseWithUsage(id, callID, name, arguments string, input, cached, cacheWrite, output int) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"completed",
		"output":[{"type":"function_call","id":"fc_1","status":"completed","call_id":%q,"name":%q,"arguments":%q}],
		"usage":{"input_tokens":%d,"input_tokens_details":{"cached_tokens":%d,"cache_write_tokens":%d},"output_tokens":%d,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":%d}
	}`, id, callID, name, arguments, input, cached, cacheWrite, output, input+output)
}

func completedTextResponseWithUsage(id, text string, input, cached, cacheWrite, output int) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":%q,"annotations":[]}]}],
		"usage":{"input_tokens":%d,"input_tokens_details":{"cached_tokens":%d,"cache_write_tokens":%d},"output_tokens":%d,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":%d}
	}`, id, text, input, cached, cacheWrite, output, input+output)
}

func combinedResponse(id, callID, name, arguments, text string) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"completed",
		"output":[
			{"type":"function_call","id":"fc_1","status":"completed","call_id":%q,"name":%q,"arguments":%q},
			{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":%q,"annotations":[]}]}
		],
		"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":6},"output_tokens":4,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":14}
	}`, id, callID, name, arguments, text)
}

func incompleteResponse(id, reason string) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"incomplete",
		"incomplete_details":{"reason":%q},
		"output":[],
		"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":2}
	}`, id, reason)
}

func refusalResponse(id, refusal string) string {
	return fmt.Sprintf(`{
		"id":%q,
		"object":"response",
		"status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":%q}]}],
		"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}
	}`, id, refusal)
}

func writeJSON(t *testing.T, w http.ResponseWriter, body string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if _, err := io.WriteString(w, body); err != nil {
		t.Fatalf("write response: %v", err)
	}
}

func writeSSE(t *testing.T, w http.ResponseWriter, event, data string) {
	t.Helper()
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data); err != nil {
		t.Fatalf("write SSE: %v", err)
	}
}

func compactJSON(t *testing.T, body string) string {
	t.Helper()
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(body)); err != nil {
		t.Fatalf("compact JSON: %v", err)
	}
	return compact.String()
}

func providerDataForTest(t *testing.T, responseID string) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(providerData{ResponseID: responseID})
	if err != nil {
		t.Fatalf("marshal provider data: %v", err)
	}
	return data
}

func messageResponseIDForTest(t *testing.T, message harness.Message) string {
	t.Helper()
	responseID, err := messageResponseID(message)
	if err != nil {
		t.Fatalf("messageResponseID() error = %v", err)
	}
	return responseID
}
