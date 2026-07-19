// Package openai implements the harness provider contract with OpenAI's
// Responses API.
package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"

	harness "github.com/lox/agent-harness"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// Provider implements harness.Provider using OpenAI's Responses API.
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

type providerData struct {
	ResponseID string `json:"response_id"`
}

// Option configures a Responses API provider.
type Option func(*config)

// WithAPIKey configures the OpenAI API key.
func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

// WithBaseURL configures a custom OpenAI endpoint.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithDefaultModel configures the default used when ChatParams.Model is empty.
func WithDefaultModel(model string) Option {
	return func(c *config) { c.defaultModel = model }
}

// WithRequestOption appends a raw openai-go request option.
func WithRequestOption(opt option.RequestOption) Option {
	return func(c *config) { c.requestOpts = append(c.requestOpts, opt) }
}

// New constructs a Responses API adapter.
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

// Chat converts harness messages and tools to Responses API input items.
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

	var terminal *responses.Response
	toolIndexes := make(map[int64]int)
	nextToolIndex := 0
	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "response.output_text.delta":
			params.OnDelta(harness.Delta{Text: event.AsResponseOutputTextDelta().Delta})
		case "response.refusal.delta":
			params.OnDelta(harness.Delta{Text: event.AsResponseRefusalDelta().Delta})
		case "response.reasoning_summary_text.delta":
			params.OnDelta(harness.Delta{Thinking: event.AsResponseReasoningSummaryTextDelta().Delta})
		case "response.output_item.added":
			added := event.AsResponseOutputItemAdded()
			if added.Item.Type == "function_call" {
				call := added.Item.AsFunctionCall()
				toolIndexes[added.OutputIndex] = nextToolIndex
				params.OnDelta(harness.Delta{ToolCall: &harness.ToolCallDelta{
					Index: nextToolIndex,
					ID:    call.CallID,
					Name:  call.Name,
				}})
				nextToolIndex++
			}
		case "response.function_call_arguments.delta":
			delta := event.AsResponseFunctionCallArgumentsDelta()
			index, ok := toolIndexes[delta.OutputIndex]
			if !ok {
				index = int(delta.OutputIndex)
			}
			params.OnDelta(harness.Delta{ToolCall: &harness.ToolCallDelta{
				Index:     index,
				Arguments: delta.Delta,
			}})
		case "response.completed":
			response := event.AsResponseCompleted().Response
			terminal = &response
		case "response.incomplete":
			response := event.AsResponseIncomplete().Response
			terminal = &response
		case "response.failed":
			response := event.AsResponseFailed().Response
			if response.Error.Message != "" {
				return nil, fmt.Errorf("responses API: %s", response.Error.Message)
			}
			return nil, fmt.Errorf("responses API response failed")
		case "error":
			streamError := event.AsError()
			return nil, fmt.Errorf("responses API: %s", streamError.Message)
		}
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	if terminal == nil {
		return nil, fmt.Errorf("responses API stream ended without a terminal response")
	}

	return convertResponse(terminal)
}

func (p *Provider) buildRequest(params harness.ChatParams) (responses.ResponseNewParams, error) {
	model := params.Model
	if model == "" {
		model = p.model
	}
	if model == "" {
		return responses.ResponseNewParams{}, fmt.Errorf("model is required")
	}

	previousResponseID, explicitContinuation := stringOption(params.Options, "previous_response_id")
	start := 0
	if explicitContinuation && previousResponseID != "" {
		var err error
		_, start, err = findResponseID(params.Messages, previousResponseID)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
	} else if !explicitContinuation {
		var err error
		previousResponseID, start, err = findResponseID(params.Messages, "")
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
	}

	instructionsBase := params.System
	for _, msg := range params.Messages[:start] {
		if msg.Role == harness.RoleSystem && msg.Content != "" {
			if instructionsBase != "" {
				instructionsBase += "\n\n"
			}
			instructionsBase += msg.Content
		}
	}
	input, instructions, err := convertMessages(instructionsBase, params.Messages[start:])
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	request := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
	}
	if instructions != "" {
		request.Instructions = param.NewOpt(instructions)
	}
	if previousResponseID != "" {
		request.PreviousResponseID = param.NewOpt(previousResponseID)
	}
	if len(params.Tools) > 0 {
		request.Tools = convertTools(params.Tools)
	}

	for key, value := range params.Options {
		switch key {
		case "previous_response_id":
			// Applied above because it also controls history conversion.
		case "prompt_cache_key":
			if v, ok := value.(string); ok {
				request.PromptCacheKey = param.NewOpt(v)
			}
		case "reasoning_effort":
			if v, ok := value.(string); ok {
				// Use the string-backed SDK type so new values such as xhigh remain
				// available even when the generated SDK constants lag the API.
				request.Reasoning.Effort = shared.ReasoningEffort(v)
			}
		case "max_output_tokens", "max_tokens":
			if v, ok := asInt64(value); ok {
				request.MaxOutputTokens = param.NewOpt(v)
			}
		case "temperature":
			if v, ok := asFloat(value); ok {
				request.Temperature = param.NewOpt(v)
			}
		case "top_p":
			if v, ok := asFloat(value); ok {
				request.TopP = param.NewOpt(v)
			}
		case "parallel_tool_calls":
			if v, ok := value.(bool); ok {
				request.ParallelToolCalls = param.NewOpt(v)
			}
		default:
			log.Printf("harness/provider/openai: ignoring unknown option %q", key)
		}
	}

	return request, nil
}

func convertMessages(system string, messages []harness.Message) (responses.ResponseInputParam, string, error) {
	input := make(responses.ResponseInputParam, 0, len(messages))
	instructions := make([]string, 0, 1)
	if system != "" {
		instructions = append(instructions, system)
	}

	for _, msg := range messages {
		switch msg.Role {
		case harness.RoleSystem:
			if msg.Content != "" {
				instructions = append(instructions, msg.Content)
			}
		case harness.RoleUser:
			input = append(input, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleUser))
		case harness.RoleAssistant:
			if msg.Content != "" {
				input = append(input, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleAssistant))
			}
			for _, call := range msg.ToolCalls {
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(call.Arguments), call.ID, call.Name))
			}
		case harness.RoleTool:
			if msg.ToolResult == nil {
				return nil, "", fmt.Errorf("tool message missing tool result")
			}
			input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(
				msg.ToolResult.ToolCallID,
				msg.ToolResult.Content,
			))
		default:
			return nil, "", fmt.Errorf("unsupported message role %q", msg.Role)
		}
	}

	return input, strings.Join(instructions, "\n\n"), nil
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
		converted := responses.ToolParamOfFunction(tool.Name, parameters, false)
		if tool.Description != "" {
			converted.OfFunction.Description = param.NewOpt(tool.Description)
		}
		out = append(out, converted)
	}
	return out
}

func convertResponse(response *responses.Response) (*harness.ChatResult, error) {
	if response == nil {
		return nil, fmt.Errorf("nil response")
	}
	switch response.Status {
	case responses.ResponseStatusCompleted, responses.ResponseStatusIncomplete:
		// Converted below.
	case responses.ResponseStatusFailed:
		if response.Error.Message != "" {
			return nil, fmt.Errorf("responses API: %s", response.Error.Message)
		}
		return nil, fmt.Errorf("responses API response failed")
	default:
		return nil, fmt.Errorf("responses API returned unexpected status %q", response.Status)
	}

	providerData, err := json.Marshal(providerData{ResponseID: response.ID})
	if err != nil {
		return nil, fmt.Errorf("encode OpenAI provider data: %w", err)
	}
	message := harness.Message{Role: harness.RoleAssistant, ProviderData: providerData}
	var textBlocks []string
	var thinking strings.Builder
	refused := false
	for _, item := range response.Output {
		switch item.Type {
		case "message":
			for _, content := range item.AsMessage().Content {
				switch content.Type {
				case "output_text":
					textBlocks = append(textBlocks, content.Text)
				case "refusal":
					refused = true
					textBlocks = append(textBlocks, content.Refusal)
				}
			}
		case "function_call":
			call := item.AsFunctionCall()
			arguments := call.Arguments
			if arguments == "" {
				arguments = `{}`
			}
			message.ToolCalls = append(message.ToolCalls, harness.ToolCall{
				ID:        call.CallID,
				Name:      call.Name,
				Arguments: json.RawMessage(arguments),
			})
		case "reasoning":
			for _, summary := range item.AsReasoning().Summary {
				thinking.WriteString(summary.Text)
			}
		}
	}
	message.Content = strings.Join(textBlocks, "\n")
	message.Thinking = thinking.String()

	finishReason := harness.FinishReasonEndTurn
	finishDetails := ""
	switch {
	case response.Status == responses.ResponseStatusIncomplete && response.IncompleteDetails.Reason == "max_output_tokens":
		finishReason = harness.FinishReasonMaxTokens
		finishDetails = response.IncompleteDetails.Reason
	case response.Status == responses.ResponseStatusIncomplete:
		finishReason = harness.FinishReasonIncomplete
		finishDetails = response.IncompleteDetails.Reason
	case refused:
		finishReason = harness.FinishReasonRefusal
		finishDetails = message.Content
	case len(message.ToolCalls) > 0:
		finishReason = harness.FinishReasonToolUse
	}

	result := &harness.ChatResult{
		Message:       message,
		ResponseID:    response.ID,
		FinishReason:  finishReason,
		FinishDetails: finishDetails,
	}
	if response.JSON.Usage.Valid() {
		result.Usage = &harness.Usage{
			InputTokens:       int(response.Usage.InputTokens),
			CachedInputTokens: int(response.Usage.InputTokensDetails.CachedTokens),
			OutputTokens:      int(response.Usage.OutputTokens),
		}
	}
	return result, nil
}

func findResponseID(messages []harness.Message, match string) (string, int, error) {
	for i := len(messages) - 1; i >= 0; i-- {
		responseID, err := messageResponseID(messages[i])
		if err != nil {
			return "", 0, err
		}
		if responseID != "" && (match == "" || responseID == match) {
			return responseID, i + 1, nil
		}
	}
	return match, 0, nil
}

func messageResponseID(message harness.Message) (string, error) {
	if message.Role != harness.RoleAssistant || len(message.ProviderData) == 0 {
		return "", nil
	}
	var data providerData
	if err := json.Unmarshal(message.ProviderData, &data); err != nil {
		return "", fmt.Errorf("decode OpenAI provider data: %w", err)
	}
	return data.ResponseID, nil
}

func stringOption(options map[string]any, key string) (string, bool) {
	if options == nil {
		return "", false
	}
	value, exists := options[key]
	if !exists {
		return "", false
	}
	text, ok := value.(string)
	return text, ok
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
