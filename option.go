package harness

import (
	"context"
	"errors"
	"fmt"
)

// Option configures the loop run.
type Option func(*options)

type options struct {
	system             string
	messages           []Message
	tools              []Tool
	model              string
	maxSteps           int
	previousResponseID string
	reasoning          ReasoningOptions
	providerOpts       map[string]any

	// Hooks
	onEvent    func(Event)
	beforeTool func(ctx context.Context, call ToolCall) (ToolAction, error)
	afterTool  func(ctx context.Context, call ToolCall, result *ToolResult) error
	onDelta    func(Delta)
	toolFilter func(step int, messages []Message) []Tool
}

func defaultOptions() options {
	return options{
		maxSteps:     10,
		providerOpts: make(map[string]any),
	}
}

func applyOptions(opts ...Option) options {
	cfg := defaultOptions()
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg
}

func (o options) validate() error {
	if o.maxSteps <= 0 {
		return errors.New("max steps must be greater than zero")
	}

	return validateTools(o.tools)
}

func validateTools(tools []Tool) error {
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return errors.New("tool name cannot be empty")
		}
		if tool.Execute == nil {
			return fmt.Errorf("tool %q has nil execute function", tool.Name)
		}
		if _, ok := seen[tool.Name]; ok {
			return fmt.Errorf("duplicate tool name %q", tool.Name)
		}
		seen[tool.Name] = struct{}{}
	}

	return nil
}

// ToolAction controls what happens after the beforeTool hook.
type ToolAction int

const (
	ToolActionContinue ToolAction = iota // execute the tool normally
	ToolActionSkip                       // skip this tool and return an error result to the LLM
	ToolActionPause                      // pause the loop and return StopPaused
)

// WithSystem sets the system prompt for the provider call.
func WithSystem(prompt string) Option {
	return func(o *options) { o.system = prompt }
}

// WithMessages sets the message history used as loop input.
func WithMessages(msgs ...Message) Option {
	return func(o *options) {
		o.messages = append([]Message(nil), msgs...)
	}
}

// WithTools sets all tools available to the loop.
func WithTools(tools ...Tool) Option {
	return func(o *options) {
		o.tools = append([]Tool(nil), tools...)
	}
}

// WithModel sets the provider model.
func WithModel(model string) Option {
	return func(o *options) { o.model = model }
}

// WithMaxSteps sets the maximum number of LLM turns.
func WithMaxSteps(n int) Option {
	return func(o *options) { o.maxSteps = n }
}

// WithPreviousResponseID continues an existing provider response on the first
// call of the run. Later calls continue from response IDs produced by the run.
func WithPreviousResponseID(id string) Option {
	return func(o *options) { o.previousResponseID = id }
}

// WithReasoning sets provider-neutral reasoning controls. Providers map the
// fields they support; generic values take precedence over compatibility keys
// in WithProviderOptions.
func WithReasoning(reasoning ReasoningOptions) Option {
	return func(o *options) { o.reasoning = reasoning }
}

// WithProviderOptions sets provider-specific options.
func WithProviderOptions(opts map[string]any) Option {
	return func(o *options) {
		o.providerOpts = copyMap(opts)
	}
}

// WithEventHandler sets a callback for loop events.
func WithEventHandler(fn func(Event)) Option {
	return func(o *options) { o.onEvent = fn }
}

// WithBeforeTool sets a callback before each tool execution.
func WithBeforeTool(fn func(ctx context.Context, call ToolCall) (ToolAction, error)) Option {
	return func(o *options) { o.beforeTool = fn }
}

// WithAfterTool sets a callback after each tool execution.
func WithAfterTool(fn func(ctx context.Context, call ToolCall, result *ToolResult) error) Option {
	return func(o *options) { o.afterTool = fn }
}

// WithOnDelta sets a callback for provider streaming deltas.
func WithOnDelta(fn func(Delta)) Option {
	return func(o *options) { o.onDelta = fn }
}

// WithToolFilter filters which tools are available on each step.
func WithToolFilter(fn func(step int, messages []Message) []Tool) Option {
	return func(o *options) { o.toolFilter = fn }
}

func emit(cfg options, e Event) {
	if cfg.onEvent != nil {
		cfg.onEvent(e)
	}
}

func copyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
