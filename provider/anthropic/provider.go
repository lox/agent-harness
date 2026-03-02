package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	harness "github.com/lox/agent-harness"
)

// Provider implements harness.Provider for Anthropic's Messages API.
type Provider struct {
	client anthropic.Client
	model  anthropic.Model
}

type config struct {
	apiKey       string
	baseURL      string
	defaultModel anthropic.Model
	requestOpts  []option.RequestOption
}

// Option configures an Anthropic provider.
type Option func(*config)

func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

func WithDefaultModel(model string) Option {
	return func(c *config) { c.defaultModel = anthropic.Model(model) }
}

func WithRequestOption(opt option.RequestOption) Option {
	return func(c *config) { c.requestOpts = append(c.requestOpts, opt) }
}

func New(opts ...Option) *Provider {
	cfg := config{defaultModel: anthropic.ModelClaudeSonnet4_20250514}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	requestOpts := make([]option.RequestOption, 0, 2+len(cfg.requestOpts))
	if cfg.apiKey != "" {
		requestOpts = append(requestOpts, option.WithAPIKey(cfg.apiKey))
	}
	if cfg.baseURL != "" {
		requestOpts = append(requestOpts, option.WithBaseURL(cfg.baseURL))
	}
	requestOpts = append(requestOpts, cfg.requestOpts...)

	return &Provider{client: anthropic.NewClient(requestOpts...), model: cfg.defaultModel}
}

func (p *Provider) Chat(ctx context.Context, params harness.ChatParams) (*harness.ChatResult, error) {
	request, err := p.buildRequest(params)
	if err != nil {
		return nil, err
	}

	if params.OnDelta == nil {
		msg, err := p.client.Messages.New(ctx, request)
		if err != nil {
			return nil, err
		}
		return convertResponse(msg)
	}

	stream := p.client.Messages.NewStreaming(ctx, request)
	defer stream.Close()

	var acc anthropic.Message
	for stream.Next() {
		event := stream.Current()
		if err := acc.Accumulate(event); err != nil {
			return nil, err
		}
		emitDeltas(params.OnDelta, event)
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}

	return convertResponse(&acc)
}

func (p *Provider) buildRequest(params harness.ChatParams) (anthropic.MessageNewParams, error) {
	model := anthropic.Model(params.Model)
	if model == "" {
		model = p.model
	}
	if model == "" {
		return anthropic.MessageNewParams{}, fmt.Errorf("model is required")
	}

	messages, systemBlocks, err := convertMessages(params.System, params.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	request := anthropic.MessageNewParams{
		Model:     model,
		Messages:  messages,
		System:    systemBlocks,
		MaxTokens: 4096,
	}

	if len(params.Tools) > 0 {
		request.Tools = convertTools(params.Tools)
	}

	for key, value := range params.Options {
		switch key {
		case "temperature":
			if v, ok := asFloat(value); ok {
				request.Temperature = anthropic.Float(v)
			}
		case "max_tokens":
			if v, ok := asInt64(value); ok {
				request.MaxTokens = v
			}
		case "top_p":
			if v, ok := asFloat(value); ok {
				request.TopP = anthropic.Float(v)
			}
		case "top_k":
			if v, ok := asInt64(value); ok {
				request.TopK = anthropic.Int(v)
			}
		case "thinking_budget":
			if v, ok := asInt64(value); ok {
				request.Thinking = anthropic.ThinkingConfigParamOfEnabled(v)
			}
		default:
			log.Printf("harness/provider/anthropic: ignoring unknown option %q", key)
		}
	}

	return request, nil
}

func convertMessages(system string, messages []harness.Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam, error) {
	out := make([]anthropic.MessageParam, 0, len(messages))
	systemBlocks := make([]anthropic.TextBlockParam, 0, 1)
	if system != "" {
		systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: system})
	}

	for _, msg := range messages {
		switch msg.Role {
		case harness.RoleSystem:
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{Text: msg.Content})
		case harness.RoleUser:
			out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)))
		case harness.RoleAssistant:
			blocks := make([]anthropic.ContentBlockParamUnion, 0, len(msg.ToolCalls)+1)
			if msg.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
			}
			for _, call := range msg.ToolCalls {
				var input any
				if len(call.Arguments) > 0 {
					if err := json.Unmarshal(call.Arguments, &input); err != nil {
						input = map[string]any{}
					}
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(call.ID, input, call.Name))
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		case harness.RoleTool:
			if msg.ToolResult == nil {
				return nil, nil, fmt.Errorf("tool message missing tool result")
			}
			out = append(out, anthropic.NewUserMessage(anthropic.NewToolResultBlock(msg.ToolResult.ToolCallID, msg.ToolResult.Content, msg.ToolResult.IsError)))
		default:
			return nil, nil, fmt.Errorf("unsupported message role %q", msg.Role)
		}
	}

	return out, systemBlocks, nil
}

func convertTools(tools []harness.ToolDef) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := anthropic.ToolInputSchemaParam{}
		if len(t.Parameters) > 0 {
			var raw map[string]any
			if err := json.Unmarshal(t.Parameters, &raw); err == nil {
				if props, ok := raw["properties"]; ok {
					schema.Properties = props
				}
				if reqRaw, ok := raw["required"].([]any); ok {
					req := make([]string, 0, len(reqRaw))
					for _, v := range reqRaw {
						if s, ok := v.(string); ok {
							req = append(req, s)
						}
					}
					schema.Required = req
				}
				schema.ExtraFields = raw
			}
		}

		tool := anthropic.ToolParam{
			Name:        t.Name,
			Description: anthropic.String(t.Description),
			InputSchema: schema,
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out
}

func emitDeltas(onDelta func(harness.Delta), event anthropic.MessageStreamEventUnion) {
	switch ev := event.AsAny().(type) {
	case anthropic.ContentBlockDeltaEvent:
		switch d := ev.Delta.AsAny().(type) {
		case anthropic.TextDelta:
			onDelta(harness.Delta{Text: d.Text})
		case anthropic.ThinkingDelta:
			onDelta(harness.Delta{Thinking: d.Thinking})
		case anthropic.InputJSONDelta:
			onDelta(harness.Delta{ToolCall: &harness.ToolCallDelta{Index: int(ev.Index), Arguments: d.PartialJSON}})
		}
	}
}

func convertResponse(msg *anthropic.Message) (*harness.ChatResult, error) {
	if msg == nil {
		return nil, fmt.Errorf("nil message")
	}

	assistant := harness.Message{Role: harness.RoleAssistant}
	for _, block := range msg.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			assistant.Content += b.Text
		case anthropic.ThinkingBlock:
			assistant.Thinking += b.Thinking
		case anthropic.ToolUseBlock:
			assistant.ToolCalls = append(assistant.ToolCalls, harness.ToolCall{ID: b.ID, Name: b.Name, Arguments: b.Input})
		}
	}

	return &harness.ChatResult{
		Message: assistant,
		Usage: &harness.Usage{
			InputTokens:  int(msg.Usage.InputTokens),
			OutputTokens: int(msg.Usage.OutputTokens),
		},
	}, nil
}

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case string:
		f, err := strconv.ParseFloat(n, 64)
		return f, err == nil
	default:
		return 0, false
	}
}

func asInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int64:
		return n, true
	case float64:
		return int64(n), true
	case json.Number:
		i, err := n.Int64()
		return i, err == nil
	case string:
		i, err := strconv.ParseInt(n, 10, 64)
		return i, err == nil
	default:
		return 0, false
	}
}

var _ harness.Provider = (*Provider)(nil)
