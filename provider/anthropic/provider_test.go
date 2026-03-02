package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	harness "github.com/lox/agent-harness"
)

func TestProviderChatNonStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("path = %q, want /v1/messages", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"claude-sonnet-4-20250514"`) {
			t.Fatalf("request body missing model: %s", body)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
		  "id":"msg_1",
		  "type":"message",
		  "role":"assistant",
		  "model":"claude-sonnet-4-20250514",
		  "content":[
		    {"type":"tool_use","id":"toolu_1","name":"echo","input":{"text":"hi"}},
		    {"type":"text","text":"hello"}
		  ],
		  "stop_reason":"tool_use",
		  "stop_sequence":null,
		  "usage":{
		    "input_tokens":11,
		    "output_tokens":5,
		    "cache_creation_input_tokens":0,
		    "cache_read_input_tokens":0
		  }
		}`))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel("claude-sonnet-4-20250514"))
	result, err := p.Chat(context.Background(), harness.ChatParams{
		System: "system prompt",
		Messages: []harness.Message{
			{Role: harness.RoleUser, Content: "hello"},
		},
		Tools: []harness.ToolDef{{
			Name:       "echo",
			Parameters: json.RawMessage(`{"type":"object"}`),
		}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.Message.Content != "hello" {
		t.Fatalf("content = %q, want hello", result.Message.Content)
	}
	if len(result.Message.ToolCalls) != 1 || result.Message.ToolCalls[0].Name != "echo" {
		t.Fatalf("tool calls = %+v", result.Message.ToolCalls)
	}
	if result.Usage == nil || result.Usage.InputTokens != 11 || result.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestProviderChatStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_start\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_delta\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n"))
		_, _ = w.Write([]byte("event: content_block_stop\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"))
		_, _ = w.Write([]byte("event: message_delta\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n"))
		_, _ = w.Write([]byte("event: message_stop\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel("claude-sonnet-4-20250514"))

	var deltas []string
	result, err := p.Chat(context.Background(), harness.ChatParams{
		Messages: []harness.Message{{Role: harness.RoleUser, Content: "hello"}},
		OnDelta: func(d harness.Delta) {
			if d.Text != "" {
				deltas = append(deltas, d.Text)
			}
		},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if !reflect.DeepEqual(deltas, []string{"Hello"}) {
		t.Fatalf("deltas = %v", deltas)
	}
	if result.Message.Content != "Hello" {
		t.Fatalf("content = %q, want Hello", result.Message.Content)
	}
}
