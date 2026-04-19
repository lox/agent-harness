package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"

	harness "github.com/lox/agent-harness"
	"github.com/lox/agent-harness/internal/conversation"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
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
		response, err := p.client.Responses.New(ctx, request)
		if err != nil {
			return nil, err
		}
		return convertResponse(response)
	}

	stream := p.client.Responses.NewStreaming(ctx, request)
	defer stream.Close()

	emitter := newStreamEmitter(params.OnDelta)
	var final *responses.Response
	for stream.Next() {
		event := stream.Current()
		emitter.emit(event)
		if completed, ok := event.AsAny().(responses.ResponseCompletedEvent); ok {
			response := completed.Response
			final = &response
		}
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	if final == nil {
		return nil, fmt.Errorf("response stream completed without a final response")
	}

	return convertResponse(final)
}

func (p *Provider) buildRequest(params harness.ChatParams) (responses.ResponseNewParams, error) {
	model := params.Model
	if model == "" {
		model = p.model
	}
	if model == "" {
		return responses.ResponseNewParams{}, fmt.Errorf("model is required")
	}

	entries, err := conversation.EntriesFromChat(params.System, params.Messages)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	request := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model),
	}

	input := convertEntries(entries)
	if len(input) > 0 {
		request.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: input}
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
				request.MaxOutputTokens = param.NewOpt(v)
			}
		case "top_p":
			if v, ok := asFloat(value); ok {
				request.TopP = param.NewOpt(v)
			}
		case "reasoning_effort":
			if v, ok := asString(value); ok {
				request.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(v)}
			}
		case "response_format":
			if v, ok := asString(value); ok {
				switch v {
				case "json_object":
					jsonFormat := shared.NewResponseFormatJSONObjectParam()
					request.Text = responses.ResponseTextConfigParam{
						Format: responses.ResponseFormatTextConfigUnionParam{OfJSONObject: &jsonFormat},
					}
				case "text":
					textFormat := shared.NewResponseFormatTextParam()
					request.Text = responses.ResponseTextConfigParam{
						Format: responses.ResponseFormatTextConfigUnionParam{OfText: &textFormat},
					}
				}
			}
		default:
			log.Printf("harness/provider/openai: ignoring unknown option %q", key)
		}
	}

	return request, nil
}

func convertEntries(entries []conversation.Entry) responses.ResponseInputParam {
	out := make(responses.ResponseInputParam, 0, len(entries))
	for _, entry := range entries {
		switch entry.Role {
		case harness.RoleSystem, harness.RoleUser, harness.RoleAssistant:
			if text := entryText(entry); text != "" {
				out = append(out, responses.ResponseInputItemParamOfMessage(text, toInputRole(entry.Role)))
			}
			if entry.Role == harness.RoleAssistant {
				for _, part := range entry.Parts {
					if part.Kind != conversation.PartToolCall {
						continue
					}
					out = append(out, responses.ResponseInputItemParamOfFunctionCall(
						normalizeArguments(part.ToolCall.Arguments),
						part.ToolCall.ID,
						part.ToolCall.Name,
					))
				}
			}
		case harness.RoleTool:
			for _, part := range entry.Parts {
				if part.Kind != conversation.PartToolResult {
					continue
				}
				out = append(out, responses.ResponseInputItemParamOfFunctionCallOutput(part.ToolResult.ToolCallID, part.ToolResult.Content))
			}
		}
	}
	return out
}

func entryText(entry conversation.Entry) string {
	var out strings.Builder
	for _, part := range entry.Parts {
		if part.Kind == conversation.PartText {
			out.WriteString(part.Text)
		}
	}
	return out.String()
}

func toInputRole(role harness.MessageRole) responses.EasyInputMessageRole {
	switch role {
	case harness.RoleSystem:
		return responses.EasyInputMessageRoleSystem
	case harness.RoleAssistant:
		return responses.EasyInputMessageRoleAssistant
	default:
		return responses.EasyInputMessageRoleUser
	}
}

func convertTools(tools []harness.ToolDef) []responses.ToolUnionParam {
	out := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		parameters := map[string]any{}
		if len(tool.Parameters) > 0 {
			if err := json.Unmarshal(tool.Parameters, &parameters); err != nil {
				parameters = map[string]any{}
			}
		}
		parameters = normalizeToolSchema(parameters)

		fn := responses.FunctionToolParam{
			Name:       tool.Name,
			Parameters: parameters,
			Strict:     param.NewOpt(true),
		}
		if tool.Description != "" {
			fn.Description = param.NewOpt(tool.Description)
		}

		out = append(out, responses.ToolUnionParam{OfFunction: &fn})
	}
	return out
}

func convertResponse(response *responses.Response) (*harness.ChatResult, error) {
	if response == nil {
		return nil, fmt.Errorf("nil response")
	}
	if len(response.Output) == 0 {
		return nil, fmt.Errorf("response had no output items")
	}

	assistant := harness.Message{Role: harness.RoleAssistant}
	var content strings.Builder
	var thinking strings.Builder

	for _, item := range response.Output {
		switch output := item.AsAny().(type) {
		case responses.ResponseOutputMessage:
			for _, part := range output.Content {
				switch contentPart := part.AsAny().(type) {
				case responses.ResponseOutputText:
					content.WriteString(contentPart.Text)
				case responses.ResponseOutputRefusal:
					content.WriteString(contentPart.Refusal)
				}
			}
		case responses.ResponseFunctionToolCall:
			callID := output.CallID
			if callID == "" {
				callID = output.ID
			}
			assistant.ToolCalls = append(assistant.ToolCalls, harness.ToolCall{
				ID:        callID,
				Name:      output.Name,
				Arguments: normalizeArgumentsJSON(output.Arguments),
			})
		case responses.ResponseReasoningItem:
			if len(output.Content) > 0 {
				for _, part := range output.Content {
					thinking.WriteString(part.Text)
				}
				continue
			}
			for _, summary := range output.Summary {
				thinking.WriteString(summary.Text)
			}
		}
	}

	assistant.Content = content.String()
	assistant.Thinking = thinking.String()

	result := &harness.ChatResult{Message: assistant}
	if response.Usage.JSON.TotalTokens.Valid() {
		result.Usage = &harness.Usage{
			InputTokens:  int(response.Usage.InputTokens),
			OutputTokens: int(response.Usage.OutputTokens),
		}
	}

	return result, nil
}

type streamEmitter struct {
	onDelta          func(harness.Delta)
	nextToolCallIdx  int
	toolCallsByIndex map[int64]streamToolCall
}

type streamToolCall struct {
	Index         int
	ID            string
	Name          string
	HeaderSent    bool
	ArgumentsSent bool
}

func newStreamEmitter(onDelta func(harness.Delta)) *streamEmitter {
	return &streamEmitter{
		onDelta:          onDelta,
		toolCallsByIndex: make(map[int64]streamToolCall),
	}
}

func (e *streamEmitter) emit(event responses.ResponseStreamEventUnion) {
	switch current := event.AsAny().(type) {
	case responses.ResponseTextDeltaEvent:
		e.onDelta(harness.Delta{Text: current.Delta})
	case responses.ResponseRefusalDeltaEvent:
		e.onDelta(harness.Delta{Text: current.Delta})
	case responses.ResponseReasoningTextDeltaEvent:
		e.onDelta(harness.Delta{Thinking: current.Delta})
	case responses.ResponseReasoningSummaryTextDeltaEvent:
		e.onDelta(harness.Delta{Thinking: current.Delta})
	case responses.ResponseOutputItemAddedEvent:
		if toolCall, ok := current.Item.AsAny().(responses.ResponseFunctionToolCall); ok {
			e.registerToolCall(current.OutputIndex, toolCall)
		}
	case responses.ResponseFunctionCallArgumentsDeltaEvent:
		e.emitToolCallDelta(current.OutputIndex, "", "", current.Delta, false)
	case responses.ResponseFunctionCallArgumentsDoneEvent:
		e.emitToolCallDelta(current.OutputIndex, "", current.Name, current.Arguments, true)
	}
}

func (e *streamEmitter) registerToolCall(outputIndex int64, toolCall responses.ResponseFunctionToolCall) {
	current, ok := e.toolCallsByIndex[outputIndex]
	if !ok {
		current.Index = e.nextToolCallIdx
		e.nextToolCallIdx++
	}
	current.ID = toolCall.CallID
	if current.ID == "" {
		current.ID = toolCall.ID
	}
	current.Name = toolCall.Name
	e.toolCallsByIndex[outputIndex] = current
}

func (e *streamEmitter) emitToolCallDelta(outputIndex int64, fallbackID, fallbackName, arguments string, final bool) {
	current, ok := e.toolCallsByIndex[outputIndex]
	if !ok {
		current = streamToolCall{Index: e.nextToolCallIdx}
		e.nextToolCallIdx++
	}

	if current.ID == "" {
		current.ID = fallbackID
	}
	if current.Name == "" {
		current.Name = fallbackName
	}

	delta := harness.ToolCallDelta{Index: current.Index}
	if !current.HeaderSent && (current.ID != "" || current.Name != "") {
		delta.ID = current.ID
		delta.Name = current.Name
		current.HeaderSent = true
	}
	if arguments != "" {
		delta.Arguments = arguments
	}

	if final && current.ArgumentsSent && delta.Arguments != "" && delta.ID == "" && delta.Name == "" {
		e.toolCallsByIndex[outputIndex] = current
		return
	}
	if delta.Arguments != "" {
		current.ArgumentsSent = true
	}

	e.toolCallsByIndex[outputIndex] = current

	if delta.Arguments == "" && delta.ID == "" && delta.Name == "" {
		return
	}
	e.onDelta(harness.Delta{ToolCall: &delta})
}

func normalizeArguments(raw json.RawMessage) string {
	if strings.TrimSpace(string(raw)) == "" {
		return "{}"
	}
	return string(raw)
}

func normalizeArgumentsJSON(raw string) json.RawMessage {
	return json.RawMessage(normalizeArguments(json.RawMessage(raw)))
}

func normalizeToolSchema(schema map[string]any) map[string]any {
	normalized, ok := normalizeSchemaValue(schema).(map[string]any)
	if !ok {
		return map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		}
	}
	return normalized
}

func normalizeSchemaValue(value any) any {
	switch v := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(v)+1)
		for key, child := range v {
			switch key {
			case "properties":
				props, ok := child.(map[string]any)
				if !ok {
					out[key] = child
					continue
				}
				normalizedProps := make(map[string]any, len(props))
				for propName, propSchema := range props {
					normalizedProps[propName] = normalizeSchemaValue(propSchema)
				}
				out[key] = normalizedProps
			case "items", "$defs", "definitions":
				out[key] = normalizeSchemaValue(child)
			case "anyOf", "allOf", "oneOf":
				out[key] = normalizeSchemaSlice(child)
			default:
				out[key] = child
			}
		}

		if looksLikeObjectSchema(out) {
			out["additionalProperties"] = false
			if _, ok := out["type"]; !ok {
				out["type"] = "object"
			}
		}
		return out
	case []any:
		out := make([]any, 0, len(v))
		for _, item := range v {
			out = append(out, normalizeSchemaValue(item))
		}
		return out
	default:
		return value
	}
}

func normalizeSchemaSlice(value any) any {
	items, ok := value.([]any)
	if !ok {
		return value
	}
	out := make([]any, 0, len(items))
	for _, item := range items {
		out = append(out, normalizeSchemaValue(item))
	}
	return out
}

func looksLikeObjectSchema(schema map[string]any) bool {
	if typ, ok := schema["type"].(string); ok && typ == "object" {
		return true
	}
	_, hasProperties := schema["properties"]
	return hasProperties
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

func asString(v any) (string, bool) {
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}
	return s, true
}
