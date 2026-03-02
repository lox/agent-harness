package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"

	harness "github.com/lox/agent-harness"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
)

// Provider implements harness.Provider using the openai-go SDK.
type Provider struct {
	client openai.Client
	model  string
}

type config struct {
	apiKey       string
	baseURL      string
	defaultModel string
	requestOpts  []option.RequestOption
}

// Option configures an OpenAI provider.
type Option func(*config)

// WithAPIKey configures the OpenAI API key.
func WithAPIKey(key string) Option {
	return func(c *config) {
		c.apiKey = key
	}
}

// WithBaseURL configures a custom OpenAI-compatible endpoint.
func WithBaseURL(url string) Option {
	return func(c *config) {
		c.baseURL = url
	}
}

// WithDefaultModel configures the provider default model used when ChatParams.Model is empty.
func WithDefaultModel(model string) Option {
	return func(c *config) {
		c.defaultModel = model
	}
}

// WithRequestOption appends a raw openai-go request option.
func WithRequestOption(opt option.RequestOption) Option {
	return func(c *config) {
		c.requestOpts = append(c.requestOpts, opt)
	}
}

// New constructs a provider/openai adapter.
func New(opts ...Option) *Provider {
	cfg := config{defaultModel: "gpt-4o-mini"}
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

	return &Provider{
		client: openai.NewClient(requestOpts...),
		model:  cfg.defaultModel,
	}
}

// Chat converts harness types to OpenAI request/response types.
func (p *Provider) Chat(ctx context.Context, params harness.ChatParams) (*harness.ChatResult, error) {
	request, err := p.buildRequest(params)
	if err != nil {
		return nil, err
	}

	if params.OnDelta == nil {
		completion, err := p.client.Chat.Completions.New(ctx, request)
		if err != nil {
			return nil, err
		}
		return convertResponse(completion)
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, request)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}
	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)
		emitDeltas(params.OnDelta, chunk)
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}

	return convertResponse(&acc.ChatCompletion)
}

func (p *Provider) buildRequest(params harness.ChatParams) (openai.ChatCompletionNewParams, error) {
	model := params.Model
	if model == "" {
		model = p.model
	}
	if model == "" {
		return openai.ChatCompletionNewParams{}, fmt.Errorf("model is required")
	}

	messages, err := convertMessages(params.System, params.Messages)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	request := openai.ChatCompletionNewParams{
		Model:    model,
		Messages: messages,
	}

	if len(params.Tools) > 0 {
		request.Tools = convertTools(params.Tools)
	}

	for key, value := range params.Options {
		switch key {
		case "temperature":
			if v, ok := asFloat(value); ok {
				request.Temperature = param.NewOpt(v)
			}
		case "max_tokens":
			if v, ok := asInt64(value); ok {
				request.MaxCompletionTokens = param.NewOpt(v)
			}
		case "top_p":
			if v, ok := asFloat(value); ok {
				request.TopP = param.NewOpt(v)
			}
		case "reasoning_effort":
			if v, ok := value.(string); ok {
				request.ReasoningEffort = shared.ReasoningEffort(v)
			}
		case "response_format":
			if v, ok := value.(string); ok {
				switch v {
				case "json_object":
					jsonFormat := shared.NewResponseFormatJSONObjectParam()
					request.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &jsonFormat}
				case "text":
					textFormat := shared.NewResponseFormatTextParam()
					request.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfText: &textFormat}
				}
			}
		default:
			log.Printf("harness/provider/openai: ignoring unknown option %q", key)
		}
	}

	return request, nil
}

func convertMessages(system string, messages []harness.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages)+1)
	if system != "" {
		out = append(out, openai.SystemMessage(system))
	}

	for _, msg := range messages {
		switch msg.Role {
		case harness.RoleSystem:
			out = append(out, openai.SystemMessage(msg.Content))
		case harness.RoleUser:
			out = append(out, openai.UserMessage(msg.Content))
		case harness.RoleAssistant:
			assistant := openai.ChatCompletionAssistantMessageParam{}
			if msg.Content != "" {
				assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{OfString: param.NewOpt(msg.Content)}
			}
			if len(msg.ToolCalls) > 0 {
				assistant.ToolCalls = make([]openai.ChatCompletionMessageToolCallParam, 0, len(msg.ToolCalls))
				for _, call := range msg.ToolCalls {
					assistant.ToolCalls = append(assistant.ToolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: call.ID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      call.Name,
							Arguments: string(call.Arguments),
						},
					})
				}
			}
			out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant})
		case harness.RoleTool:
			if msg.ToolResult == nil {
				return nil, fmt.Errorf("tool message missing tool result")
			}
			out = append(out, openai.ToolMessage(msg.ToolResult.Content, msg.ToolResult.ToolCallID))
		default:
			return nil, fmt.Errorf("unsupported message role %q", msg.Role)
		}
	}

	return out, nil
}

func convertTools(tools []harness.ToolDef) []openai.ChatCompletionToolParam {
	out := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		parameters := shared.FunctionParameters{}
		if len(tool.Parameters) > 0 {
			if err := json.Unmarshal(tool.Parameters, &parameters); err != nil {
				parameters = shared.FunctionParameters{}
			}
		}
		out = append(out, openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters:  parameters,
			},
		})
	}
	return out
}

func emitDeltas(onDelta func(harness.Delta), chunk openai.ChatCompletionChunk) {
	if len(chunk.Choices) == 0 {
		return
	}
	delta := chunk.Choices[0].Delta
	if delta.Content != "" {
		onDelta(harness.Delta{Text: delta.Content})
	}
	for _, tc := range delta.ToolCalls {
		onDelta(harness.Delta{ToolCall: &harness.ToolCallDelta{
			Index:     int(tc.Index),
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		}})
	}
}

func convertResponse(completion *openai.ChatCompletion) (*harness.ChatResult, error) {
	if completion == nil {
		return nil, fmt.Errorf("nil completion")
	}
	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("completion had no choices")
	}

	choice := completion.Choices[0]
	assistant := harness.Message{
		Role:    harness.RoleAssistant,
		Content: choice.Message.Content,
	}

	if len(choice.Message.ToolCalls) > 0 {
		assistant.ToolCalls = make([]harness.ToolCall, 0, len(choice.Message.ToolCalls))
		for _, toolCall := range choice.Message.ToolCalls {
			assistant.ToolCalls = append(assistant.ToolCalls, harness.ToolCall{
				ID:        toolCall.ID,
				Name:      toolCall.Function.Name,
				Arguments: json.RawMessage(toolCall.Function.Arguments),
			})
		}
	}

	result := &harness.ChatResult{Message: assistant}
	if completion.Usage.JSON.TotalTokens.Valid() {
		result.Usage = &harness.Usage{
			InputTokens:  int(completion.Usage.PromptTokens),
			OutputTokens: int(completion.Usage.CompletionTokens),
		}
	}

	return result, nil
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
