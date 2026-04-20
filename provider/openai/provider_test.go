package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	harness "github.com/lox/agent-harness"
	sdkopenai "github.com/openai/openai-go/v3"
	sdkresponses "github.com/openai/openai-go/v3/responses"
	sdkshared "github.com/openai/openai-go/v3/shared"
)

func TestBuildRequestConvertsConversationToResponsesInput(t *testing.T) {
	request, err := New().buildRequest(harness.ChatParams{
		Model:  string(sdkopenai.ChatModelGPT5_4Mini),
		System: "system prompt",
		Messages: []harness.Message{
			{Role: harness.RoleUser, Content: "hello"},
			{
				Role:    harness.RoleAssistant,
				Content: "working",
				ToolCalls: []harness.ToolCall{{
					ID:        "call_1",
					Name:      "echo",
					Arguments: json.RawMessage(`{"text":"hi"}`),
				}},
			},
			{Role: harness.RoleTool, ToolResult: &harness.ToolResult{ToolCallID: "call_1", Content: "tool said hi"}},
		},
	})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}

	items := request.Input.OfInputItemList
	if len(items) != 5 {
		t.Fatalf("input items = %d, want 5", len(items))
	}
	if got := items[0].OfMessage.Role; got != sdkresponses.EasyInputMessageRoleSystem {
		t.Fatalf("item 0 role = %q, want system", got)
	}
	if got := items[0].OfMessage.Content.OfString.Value; got != "system prompt" {
		t.Fatalf("item 0 content = %q, want system prompt", got)
	}
	if got := items[1].OfMessage.Role; got != sdkresponses.EasyInputMessageRoleUser {
		t.Fatalf("item 1 role = %q, want user", got)
	}
	if got := items[2].OfMessage.Role; got != sdkresponses.EasyInputMessageRoleAssistant {
		t.Fatalf("item 2 role = %q, want assistant", got)
	}
	if got := items[2].OfMessage.Content.OfString.Value; got != "working" {
		t.Fatalf("item 2 content = %q, want working", got)
	}
	if got := items[3].OfFunctionCall.CallID; got != "call_1" {
		t.Fatalf("item 3 call id = %q, want call_1", got)
	}
	if got := items[3].OfFunctionCall.Arguments; got != `{"text":"hi"}` {
		t.Fatalf("item 3 arguments = %q, want tool args", got)
	}
	if got := items[4].OfFunctionCallOutput.CallID; got != "call_1" {
		t.Fatalf("item 4 call id = %q, want call_1", got)
	}
	if got := items[4].OfFunctionCallOutput.Output.OfString.Value; got != "tool said hi" {
		t.Fatalf("item 4 output = %q, want tool said hi", got)
	}
}

func TestBuildRequestNormalizesToolSchemasForStrictMode(t *testing.T) {
	request, err := New().buildRequest(harness.ChatParams{
		Model:    string(sdkopenai.ChatModelGPT5_4Mini),
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		Tools: []harness.ToolDef{{
			Name: "echo",
			Parameters: json.RawMessage(`{
				"type":"object",
				"properties":{
					"text":{"type":"string"},
					"options":{
						"type":"object",
						"properties":{"loud":{"type":"boolean"}}
					}
				},
				"required":["text"]
			}`),
		}},
	})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}

	schema := request.Tools[0].OfFunction.Parameters
	if got, ok := schema["additionalProperties"].(bool); !ok || got {
		t.Fatalf("root additionalProperties = %#v, want false", schema["additionalProperties"])
	}

	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties type = %T, want map[string]any", schema["properties"])
	}
	options, ok := props["options"].(map[string]any)
	if !ok {
		t.Fatalf("options type = %T, want map[string]any", props["options"])
	}
	if got, ok := options["additionalProperties"].(bool); !ok || got {
		t.Fatalf("nested additionalProperties = %#v, want false", options["additionalProperties"])
	}
}

func TestBuildRequestNormalizesEmptyToolSchemaForStrictMode(t *testing.T) {
	request, err := New().buildRequest(harness.ChatParams{
		Model:    string(sdkopenai.ChatModelGPT5_4Mini),
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		Tools: []harness.ToolDef{{
			Name: "time_now",
		}},
	})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}

	schema := request.Tools[0].OfFunction.Parameters
	if got, ok := schema["type"].(string); !ok || got != "object" {
		t.Fatalf("type = %#v, want object", schema["type"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties type = %T, want map[string]any", schema["properties"])
	}
	if len(props) != 0 {
		t.Fatalf("properties len = %d, want 0", len(props))
	}
	if got, ok := schema["additionalProperties"].(bool); !ok || got {
		t.Fatalf("additionalProperties = %#v, want false", schema["additionalProperties"])
	}
}

func TestProviderChatNonStreamingUsesResponses(t *testing.T) {
	model := string(sdkopenai.ChatModelGPT5_4)
	reasoningEffort := string(sdkshared.ReasoningEffortXhigh)

	var called int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&called, 1)
		if r.URL.Path != "/responses" {
			t.Fatalf("path = %q, want /responses", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		bodyStr := string(body)
		if !strings.Contains(bodyStr, `"model":"`+model+`"`) {
			t.Fatalf("request body missing model: %s", body)
		}
		if !strings.Contains(bodyStr, `"reasoning":{"effort":"`+reasoningEffort+`"}`) {
			t.Fatalf("request body missing reasoning effort: %s", body)
		}
		if !strings.Contains(bodyStr, `"tools":[`) {
			t.Fatalf("request body missing tools: %s", body)
		}
		if !strings.Contains(bodyStr, `"additionalProperties":false`) {
			t.Fatalf("request body missing normalized strict schema: %s", body)
		}
		if !strings.Contains(bodyStr, `"type":"function_call"`) || !strings.Contains(bodyStr, `"call_id":"call_prev"`) {
			t.Fatalf("request body missing prior function call: %s", body)
		}
		if !strings.Contains(bodyStr, `"type":"function_call_output"`) || !strings.Contains(bodyStr, `"output":"tool said hi"`) {
			t.Fatalf("request body missing function call output: %s", body)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(fmt.Sprintf(`{
		  "id":"resp_1",
		  "object":"response",
		  "created_at":1,
		  "model":"%s",
		  "status":"completed",
		  "output":[{
		    "id":"fc_1",
		    "type":"function_call",
		    "call_id":"call_1",
		    "name":"echo",
		    "arguments":"{\"text\":\"hi\"}",
		    "status":"completed"
		  }],
		  "usage":{
		    "input_tokens":12,
		    "input_tokens_details":{"cached_tokens":0},
		    "output_tokens":5,
		    "output_tokens_details":{"reasoning_tokens":2},
		    "total_tokens":17
		  }
		}`, model)))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel(model))
	result, err := p.Chat(context.Background(), harness.ChatParams{
		System: "system prompt",
		Messages: []harness.Message{
			{Role: harness.RoleUser, Content: "hello"},
			{
				Role: harness.RoleAssistant,
				ToolCalls: []harness.ToolCall{{
					ID:        "call_prev",
					Name:      "echo",
					Arguments: json.RawMessage(`{"text":"before"}`),
				}},
			},
			{Role: harness.RoleTool, ToolResult: &harness.ToolResult{ToolCallID: "call_prev", Content: "tool said hi"}},
		},
		Tools: []harness.ToolDef{{
			Name:        "echo",
			Description: "Echo input",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		}},
		Options: map[string]any{
			"reasoning_effort": reasoningEffort,
		},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if atomic.LoadInt32(&called) != 1 {
		t.Fatalf("server call count = %d, want 1", called)
	}
	if result.Message.Role != harness.RoleAssistant {
		t.Fatalf("role = %q", result.Message.Role)
	}
	if len(result.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(result.Message.ToolCalls))
	}
	if result.Message.ToolCalls[0].ID != "call_1" {
		t.Fatalf("tool call id = %q, want call_1", result.Message.ToolCalls[0].ID)
	}
	if result.Message.ToolCalls[0].Name != "echo" {
		t.Fatalf("tool name = %q, want echo", result.Message.ToolCalls[0].Name)
	}
	if result.Usage == nil || result.Usage.InputTokens != 12 || result.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestProviderChatStreamingTextAndThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Used concise \",\"item_id\":\"reason_1\",\"output_index\":0,\"summary_index\":0,\"sequence_number\":1}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"reasoning\",\"item_id\":\"reason_1\",\"output_index\":0,\"summary_index\":0,\"sequence_number\":2}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\",\"item_id\":\"msg_1\",\"output_index\":1,\"content_index\":0,\"sequence_number\":3,\"logprobs\":[]}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\",\"item_id\":\"msg_1\",\"output_index\":1,\"content_index\":0,\"sequence_number\":4,\"logprobs\":[]}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"sequence_number\":5,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"gpt-5.4-mini\",\"status\":\"completed\",\"output\":[{\"id\":\"reason_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Used concise reasoning\"}],\"status\":\"completed\"},{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":2,\"output_tokens_details\":{\"reasoning_tokens\":1},\"total_tokens\":9}}}\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel(string(sdkopenai.ChatModelGPT5_4Mini)))

	var textDeltas []string
	var thinkingDeltas []string
	result, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta: func(d harness.Delta) {
			if d.Text != "" {
				textDeltas = append(textDeltas, d.Text)
			}
			if d.Thinking != "" {
				thinkingDeltas = append(thinkingDeltas, d.Thinking)
			}
		},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got := strings.Join(textDeltas, ""); got != "Hello" {
		t.Fatalf("text deltas = %q, want Hello", got)
	}
	if got := strings.Join(thinkingDeltas, ""); got != "Used concise reasoning" {
		t.Fatalf("thinking deltas = %q, want Used concise reasoning", got)
	}
	if result.Message.Content != "Hello" {
		t.Fatalf("content = %q, want Hello", result.Message.Content)
	}
	if result.Message.Thinking != "Used concise reasoning" {
		t.Fatalf("thinking = %q, want Used concise reasoning", result.Message.Thinking)
	}
}

func TestProviderChatStreamingToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"text\\\":\\\"\",\"item_id\":\"fc_1\",\"output_index\":0,\"sequence_number\":2}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"hi\\\"}\",\"item_id\":\"fc_1\",\"output_index\":0,\"sequence_number\":3}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"text\\\":\\\"hi\\\"}\",\"item_id\":\"fc_1\",\"name\":\"echo\",\"output_index\":0,\"sequence_number\":4}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"sequence_number\":5,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"gpt-5.4-mini\",\"status\":\"completed\",\"output\":[{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"{\\\"text\\\":\\\"hi\\\"}\",\"status\":\"completed\"}],\"usage\":{\"input_tokens\":9,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":3,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":12}}}\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel(string(sdkopenai.ChatModelGPT5_4Mini)))

	var deltas []harness.ToolCallDelta
	result, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta: func(d harness.Delta) {
			if d.ToolCall != nil {
				deltas = append(deltas, *d.ToolCall)
			}
		},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(deltas) != 2 {
		t.Fatalf("tool deltas = %d, want 2", len(deltas))
	}
	if deltas[0].Index != 0 || deltas[0].ID != "call_1" || deltas[0].Name != "echo" || deltas[0].Arguments != "{\"text\":\"" {
		t.Fatalf("first delta = %+v", deltas[0])
	}
	if deltas[1].Index != 0 || deltas[1].ID != "" || deltas[1].Name != "" || deltas[1].Arguments != "hi\"}" {
		t.Fatalf("second delta = %+v", deltas[1])
	}
	if len(result.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(result.Message.ToolCalls))
	}
	if result.Message.ToolCalls[0].ID != "call_1" {
		t.Fatalf("tool call id = %q, want call_1", result.Message.ToolCalls[0].ID)
	}
	if string(result.Message.ToolCalls[0].Arguments) != `{"text":"hi"}` {
		t.Fatalf("tool call arguments = %s, want {\"text\":\"hi\"}", result.Message.ToolCalls[0].Arguments)
	}
}

func TestProviderChatStreamingAcceptsIncompleteTerminalResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"sequence_number\":1,\"logprobs\":[]}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"sequence_number\":2,\"logprobs\":[]}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.incomplete\",\"sequence_number\":3,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"gpt-5.4-mini\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"incomplete\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":2,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":9}}}\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel(string(sdkopenai.ChatModelGPT5_4Mini)))

	var textDeltas []string
	result, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta: func(d harness.Delta) {
			if d.Text != "" {
				textDeltas = append(textDeltas, d.Text)
			}
		},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got := strings.Join(textDeltas, ""); got != "Hello" {
		t.Fatalf("text deltas = %q, want Hello", got)
	}
	if result.Message.Content != "Hello" {
		t.Fatalf("content = %q, want Hello", result.Message.Content)
	}
}

func TestProviderChatStreamingReturnsResponseFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.failed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"gpt-5.4-mini\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"backend blew up\"},\"output\":[]}}\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel(string(sdkopenai.ChatModelGPT5_4Mini)))

	_, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta:  func(harness.Delta) {},
	})
	if err == nil {
		t.Fatal("Chat() error = nil, want response failure")
	}
	if got := err.Error(); got != "response failed: backend blew up" {
		t.Fatalf("Chat() error = %q, want response failed: backend blew up", got)
	}
}

func TestProviderChatStreamingReturnsResponseErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"error\",\"sequence_number\":1,\"code\":\"server_error\",\"message\":\"gateway exploded\",\"param\":\"\"}\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel(string(sdkopenai.ChatModelGPT5_4Mini)))

	_, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta:  func(harness.Delta) {},
	})
	if err == nil {
		t.Fatal("Chat() error = nil, want response error event failure")
	}
	if got := err.Error(); got != "response error: gateway exploded" {
		t.Fatalf("Chat() error = %q, want response error: gateway exploded", got)
	}
}

func TestProviderChatIgnoresBlankReasoningEffort(t *testing.T) {
	request, err := New().buildRequest(harness.ChatParams{
		Model:    string(sdkopenai.ChatModelGPT5_4),
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		Options: map[string]any{
			"reasoning_effort": "   ",
		},
	})
	if err != nil {
		t.Fatalf("buildRequest() error = %v", err)
	}
	if request.Reasoning.Effort != "" {
		t.Fatalf("reasoning effort = %q, want empty", request.Reasoning.Effort)
	}
}

func TestConvertResponseNormalizesEmptyToolArguments(t *testing.T) {
	var response sdkresponses.Response
	if err := json.Unmarshal([]byte(`{
		"id":"resp_1",
		"object":"response",
		"created_at":1,
		"model":"gpt-5.4-mini",
		"status":"completed",
		"output":[
			{
				"id":"fc_1",
				"type":"function_call",
				"call_id":"call_1",
				"name":"time_now",
				"arguments":"",
				"status":"completed"
			}
		],
		"usage":{
			"input_tokens":1,
			"input_tokens_details":{"cached_tokens":0},
			"output_tokens":1,
			"output_tokens_details":{"reasoning_tokens":0},
			"total_tokens":2
		}
	}`), &response); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	result, err := convertResponse(&response)
	if err != nil {
		t.Fatalf("convertResponse() error = %v", err)
	}
	if len(result.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(result.Message.ToolCalls))
	}
	if string(result.Message.ToolCalls[0].Arguments) != "{}" {
		t.Fatalf("tool arguments = %q, want {}", result.Message.ToolCalls[0].Arguments)
	}
}
