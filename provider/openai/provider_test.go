package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	harness "github.com/lox/agent-harness"
)

func TestProviderChatNonStreaming(t *testing.T) {
	var called int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&called, 1)
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %q, want /chat/completions", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"gpt-4o"`) {
			t.Fatalf("request body missing model: %s", body)
		}
		if !strings.Contains(string(body), `"tools"`) {
			t.Fatalf("request body missing tools: %s", body)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
		  "id":"chatcmpl-1",
		  "object":"chat.completion",
		  "created":1,
		  "model":"gpt-4o",
		  "choices":[{
		    "index":0,
		    "finish_reason":"tool_calls",
		    "message":{
		      "role":"assistant",
		      "content":"",
		      "tool_calls":[{
		        "id":"call_1",
		        "type":"function",
		        "function":{"name":"echo","arguments":"{\"text\":\"hi\"}"}
		      }]
		    }
		  }],
		  "usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}
		}`))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel("gpt-4o"))
	result, err := p.Chat(context.Background(), harness.ChatParams{
		System: "system prompt",
		Messages: []harness.Message{
			{Role: harness.RoleUser, Content: "hello"},
		},
		Tools: []harness.ToolDef{{
			Name:        "echo",
			Description: "Echo input",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		}},
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
	if result.Message.ToolCalls[0].Name != "echo" {
		t.Fatalf("tool name = %q", result.Message.ToolCalls[0].Name)
	}
	if result.Usage == nil || result.Usage.InputTokens != 12 || result.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestProviderChatStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(WithBaseURL(server.URL), WithAPIKey("test-key"), WithDefaultModel("gpt-4o"))

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
	if !reflect.DeepEqual(deltas, []string{"Hel", "lo"}) {
		t.Fatalf("deltas = %#v", deltas)
	}
	if result.Message.Content != "Hello" {
		t.Fatalf("content = %q, want Hello", result.Message.Content)
	}
}
