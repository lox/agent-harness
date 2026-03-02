# agent-harness — Go Agent Loop Library

A minimal, composable Go library for building agentic tool-calling loops on top of LLM APIs.

## Motivation

Every agent project (Pincer, internal tools, CLI agents) re-implements the same core loop: call the LLM, parse tool calls, execute tools, feed results back, repeat. The loop logic gets entangled with application concerns (approval gates, persistence, event emission, prompt construction). This library extracts the universal parts.

## Design Principles

1. **Small API surface** — a single `Run()` function and a handful of interfaces, not a framework
2. **Bring your own provider** — the library doesn't bundle HTTP clients for Anthropic/OpenAI/etc; you implement a simple interface
3. **Bring your own tools** — tools are just functions with a JSON schema; no registry magic
4. **Hooks over inheritance** — customise behaviour via callbacks (before/after tool execution, on message, on error), not by subclassing
5. **Streaming optional** — works with both streaming and non-streaming providers
6. **Context-first** — all operations respect `context.Context` for cancellation and timeouts
7. **No global state** — everything is passed explicitly; safe for concurrent use

## Research Summary

### Pi (TypeScript)
- **Agent loop** is a pure async generator yielding events, with two nested loops (inner: tool calls, outer: follow-ups/steering)
- **EventStream** abstraction: async iterable that yields events and resolves to a final value
- **Steering/follow-up queues** let external code inject messages mid-loop (steering interrupts tool batches, follow-ups extend after completion)
- **Tool results** include both content (for LLM) and details (for UI)
- **Message types** are extensible via declaration merging — custom message types are converted to LLM-compatible format via a `convertToLlm` function
- **Provider abstraction** is a registry of `streamSimple` functions keyed by API name

### Picoclaw (Go)
- **`LLMProvider` interface** is minimal: single `Chat(ctx, messages, tools, model, opts) (*Response, error)` method
- **`RunToolLoop`** is extracted as a standalone reusable function — both the main agent and subagents call it
- **`ToolResult`** has dual outputs: `ForLLM` (fed back to model) and `ForUser` (shown to human)
- **Tool interfaces** are layered: base `Tool`, optional `ContextualTool`, optional `AsyncTool` — detected via type assertion
- **Fallback chain** with error classification (auth, rate limit, timeout, format) drives retry/fallback decisions
- **Stable tool ordering** for LLM KV cache efficiency
- **Context compression** when history exceeds token budget (summarise + truncate)

### Pincer (Go) — Target Consumer
- **230+ line `executeTurnFromStep`** mixes loop logic, tool classification, persistence, events, and approval policy
- **Tool schemas** in one file, **tool execution** in another via `switch` statement — adding tools requires editing 3+ places
- **Approval gate** pauses the loop when non-READ tools are proposed; resumes via work queue on approval
- **No streaming** in the planner; full response fetched then parsed
- **Step budget** tracked in 3 different places across continuations
- Pain points: tight coupling, fragmented tool registration, hand-rolled message format, complex resume/continuation logic

## Architecture

```
agent-harness/
├── provider.go      # LLM provider interface
├── tool.go          # Tool interface and types
├── message.go       # Message types (user, assistant, tool_call, tool_result)
├── thread.go        # Thread container for conversation state
├── loop.go          # The core Run() function
├── event.go         # Event types yielded during the loop
├── option.go        # Functional options for Run()
└── provider/        # Optional: reference provider implementations
    ├── anthropic/
    └── openai/
```

## Core Types

### Messages

Each message represents exactly one logical payload type. Tool results are
separate messages (one per tool call) with `RoleTool`, which maps directly
to OpenAI's wire format and is trivially adapted to Anthropic's content
blocks by the provider adapter.

```go
// MessageRole identifies the role of a message in the conversation.
type MessageRole string

const (
    RoleUser      MessageRole = "user"
    RoleAssistant MessageRole = "assistant"
    RoleSystem    MessageRole = "system"
    RoleTool      MessageRole = "tool"
)

// Message is a single message in the conversation history.
// Each message has exactly one of: text content, tool calls, or a tool result.
type Message struct {
    Role    MessageRole
    Content string

    // For RoleAssistant messages that include tool calls.
    ToolCalls []ToolCall

    // For RoleTool messages — the result of a single tool call.
    ToolResult *ToolResult

    // Thinking/reasoning content from the model (if any).
    Thinking string
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
    ID        string
    Name      string
    Arguments json.RawMessage // unparsed JSON arguments
}
```

### Provider Interface

Inspired by Picoclaw's minimal approach. One method. Streaming is handled via a callback option rather than a separate interface.

```go
// Provider sends messages to an LLM and returns a response.
type Provider interface {
    // Chat sends the conversation to the model and returns the response.
    // tools may be nil if no tools are available for this call.
    Chat(ctx context.Context, params ChatParams) (*ChatResult, error)
}

// ChatParams contains everything needed for an LLM call.
type ChatParams struct {
    Model    string
    System   string          // system prompt
    Messages []Message       // conversation history
    Tools    []ToolDef       // available tool definitions
    Options  map[string]any  // provider-specific options (temperature, max_tokens, etc.)

    // Optional: called with streaming deltas as they arrive.
    // If nil, the provider should still return the complete response.
    OnDelta func(Delta)
}

// ChatResult is what the provider returns.
type ChatResult struct {
    Message Message    // the assistant's response (may include tool calls)
    Usage   *Usage     // token usage, if available
}

// Usage tracks token consumption.
type Usage struct {
    InputTokens  int
    OutputTokens int
}

// Delta represents a streaming chunk from the provider.
type Delta struct {
    Text      string // text content delta
    Thinking  string // thinking/reasoning delta
    ToolCall  *ToolCallDelta
}

type ToolCallDelta struct {
    Index     int
    ID        string // only set on first delta for this tool call
    Name      string // only set on first delta for this tool call
    Arguments string // JSON fragment
}
```

### Tool Interface

Inspired by both Pi and Picoclaw. Tools bundle their schema and execution together. The `ForLLM`/`ForUser` split from Picoclaw is valuable.

```go
// ToolDef describes a tool's schema for the LLM.
type ToolDef struct {
    Name        string
    Description string
    Parameters  json.RawMessage // JSON Schema object
}

// Tool is a callable tool that the agent can use.
type Tool struct {
    ToolDef

    // Execute runs the tool with the given arguments.
    // The arguments are the raw JSON from the model's tool call.
    Execute func(ctx context.Context, call ToolCall) (*ToolResult, error)
}

// ToolResult is the output of a tool execution.
type ToolResult struct {
    ToolCallID string

    // Content is sent back to the LLM as the tool's response.
    Content string

    // UserContent is optionally shown to the user instead of/in addition to Content.
    // If empty, Content is used for both purposes.
    UserContent string

    // IsError indicates the tool execution failed.
    IsError bool

    // Metadata is arbitrary structured data for the caller (not sent to LLM).
    Metadata map[string]any
}
```

### The Loop

The core of the library. A single `Run` function that implements the tool-calling loop.

```go
// Run executes the agent loop: call the LLM, execute any tool calls,
// feed results back, repeat until the model stops calling tools or
// the step budget is exhausted.
//
// It returns the complete list of new messages generated during this run
// (assistant messages, tool results, etc.).
func Run(ctx context.Context, provider Provider, opts ...Option) (*Result, error)

// Result contains everything produced by a single Run invocation.
type Result struct {
    // Messages contains all messages generated during this run,
    // in order. Includes assistant messages and tool result messages.
    // When StopReason is StopPaused, these are the messages completed
    // before the pause — they do NOT include the pending tool calls.
    Messages []Message

    // TotalUsage is the sum of all LLM calls made during this run.
    TotalUsage Usage

    // Steps is the number of LLM calls made (1 = no tool calls).
    Steps int

    // StopReason indicates why the loop terminated.
    StopReason StopReason

    // PendingToolCalls contains tool calls that were proposed by the model
    // but not yet executed when the loop was paused. Only populated when
    // StopReason is StopPaused. The caller is responsible for executing
    // these (after approval, etc.) and providing the results as RoleTool
    // messages when resuming.
    PendingToolCalls []ToolCall
}

type StopReason int

const (
    StopEndTurn    StopReason = iota // model finished naturally (no more tool calls)
    StopMaxSteps                     // step budget exhausted
    StopCancelled                    // context cancelled
    StopPaused                       // hook requested pause (e.g. for approval)
)
```

### Options (Functional Options Pattern)

```go
type options struct {
    system      string
    messages    []Message
    tools       []Tool
    model       string
    maxSteps    int
    providerOpts map[string]any

    // Hooks
    onEvent     func(Event)
    beforeTool  func(ctx context.Context, call ToolCall) (ToolAction, error)
    afterTool   func(ctx context.Context, call ToolCall, result *ToolResult) error
    onDelta     func(Delta)
    toolFilter  func(step int, messages []Message) []Tool
}

// ToolAction controls what happens after the beforeTool hook.
type ToolAction int

const (
    ToolActionContinue ToolAction = iota // execute the tool normally
    ToolActionSkip                       // skip this tool, return an error result to the LLM
    ToolActionPause                      // pause the loop, return StopPaused
)

func WithSystem(prompt string) Option
func WithMessages(msgs ...Message) Option
func WithTools(tools ...Tool) Option
func WithModel(model string) Option
func WithMaxSteps(n int) Option
func WithProviderOptions(opts map[string]any) Option

// Hooks
func WithEventHandler(fn func(Event)) Option
func WithBeforeTool(fn func(ctx context.Context, call ToolCall) (ToolAction, error)) Option
func WithAfterTool(fn func(ctx context.Context, call ToolCall, result *ToolResult) error) Option
func WithOnDelta(fn func(Delta)) Option

// Progressive disclosure: filter which tools are available per step.
// Called before each LLM call. If nil, all tools from WithTools are used.
func WithToolFilter(fn func(step int, messages []Message) []Tool) Option
```

### Events

```go
type EventType int

const (
    EventTurnStart       EventType = iota // LLM call starting
    EventTurnEnd                          // LLM call completed
    EventToolStart                        // tool execution starting
    EventToolEnd                          // tool execution completed
    EventMessage                          // new message added to history
    EventError                            // non-fatal error (e.g. tool failure)
)

type Event struct {
    Type     EventType
    Step     int          // which step of the loop (0-indexed)
    Message  *Message     // for EventMessage
    ToolCall *ToolCall    // for EventToolStart/EventToolEnd
    Result   *ToolResult  // for EventToolEnd
    Error    error        // for EventError
}
```

## Loop Pseudocode

```go
func Run(ctx context.Context, provider Provider, opts ...Option) (*Result, error) {
    cfg := applyOptions(opts)
    if err := cfg.validate(); err != nil {
        return nil, err
    }

    messages := slices.Clone(cfg.messages)
    toolMap := buildToolMap(cfg.tools) // map[string]*Tool for O(1) lookup
    var result Result

    for step := 0; step < cfg.maxSteps; step++ {
        emit(cfg, Event{Type: EventTurnStart, Step: step})

        // Progressive disclosure: filter tools per step
        activeTools := cfg.tools
        if cfg.toolFilter != nil {
            activeTools = cfg.toolFilter(step, messages)
            toolMap = buildToolMap(activeTools)
        }

        // Call the LLM
        chatResult, err := provider.Chat(ctx, ChatParams{
            Model:    cfg.model,
            System:   cfg.system,
            Messages: messages,
            Tools:    toolDefs(activeTools),
            Options:  cfg.providerOpts,
            OnDelta:  cfg.onDelta,
        })
        if err != nil {
            return nil, fmt.Errorf("step %d: %w", step, err)
        }

        // Append assistant message
        assistantMsg := chatResult.Message
        messages = append(messages, assistantMsg)
        result.Messages = append(result.Messages, assistantMsg)
        result.TotalUsage.Add(chatResult.Usage)
        result.Steps = step + 1
        emit(cfg, Event{Type: EventMessage, Step: step, Message: assistantMsg})
        emit(cfg, Event{Type: EventTurnEnd, Step: step})

        // No tool calls? We're done.
        if len(assistantMsg.ToolCalls) == 0 {
            result.StopReason = StopEndTurn
            return &result, nil
        }

        // Execute tool calls — one RoleTool message per call
        var pendingCalls []ToolCall

        for i, call := range assistantMsg.ToolCalls {
            // Before-tool hook (for approval gates, logging, etc.)
            if cfg.beforeTool != nil {
                action, err := cfg.beforeTool(ctx, call)
                if err != nil {
                    return nil, err
                }
                switch action {
                case ToolActionSkip:
                    toolMsg := Message{
                        Role: RoleTool,
                        ToolResult: &ToolResult{
                            ToolCallID: call.ID,
                            Content:    "Tool call skipped.",
                            IsError:    true,
                        },
                    }
                    messages = append(messages, toolMsg)
                    result.Messages = append(result.Messages, toolMsg)
                    continue
                case ToolActionPause:
                    // Collect this and all remaining calls as pending
                    pendingCalls = assistantMsg.ToolCalls[i:]
                    break
                }
            }
            if len(pendingCalls) > 0 {
                break
            }

            // Find and execute the tool
            emit(cfg, Event{Type: EventToolStart, Step: step, ToolCall: &call})

            var toolResult *ToolResult
            tool, ok := toolMap[call.Name]
            if !ok {
                toolResult = &ToolResult{
                    ToolCallID: call.ID,
                    Content:    fmt.Sprintf("Unknown tool: %s", call.Name),
                    IsError:    true,
                }
            } else {
                toolResult, err = tool.Execute(ctx, call)
                if err != nil {
                    // Sanitise: don't leak error details to the model
                    toolResult = &ToolResult{
                        ToolCallID: call.ID,
                        Content:    "Tool execution failed.",
                        IsError:    true,
                        Metadata:   map[string]any{"error": err.Error()},
                    }
                }
            }

            if cfg.afterTool != nil {
                cfg.afterTool(ctx, call, toolResult)
            }

            emit(cfg, Event{Type: EventToolEnd, Step: step, ToolCall: &call, Result: toolResult})

            // Append as a separate RoleTool message
            toolMsg := Message{Role: RoleTool, ToolResult: toolResult}
            messages = append(messages, toolMsg)
            result.Messages = append(result.Messages, toolMsg)
            emit(cfg, Event{Type: EventMessage, Step: step, Message: toolMsg})
        }

        // If we paused, return pending calls without appending incomplete results
        if len(pendingCalls) > 0 {
            result.StopReason = StopPaused
            result.PendingToolCalls = pendingCalls
            return &result, nil
        }
    }

    result.StopReason = StopMaxSteps
    return &result, nil
}
```

## Thread Management

The harness is stateless — it takes messages in and returns new messages out.
The caller owns storage. The `Thread` type is a thin container that formalises
the state callers need to track between runs, particularly `PendingToolCalls`
for pause/resume workflows. No `Store` interface — everyone's backend is
different.

```go
// Thread tracks conversation state between runs. The caller is responsible
// for serializing this to their storage backend (SQLite, file, etc.).
type Thread struct {
    ID               string         `json:"id"`
    Messages         []Message      `json:"messages"`
    PendingToolCalls []ToolCall     `json:"pending_tool_calls,omitempty"`
    Metadata         map[string]any `json:"metadata,omitempty"`
}

// NewThread creates a thread with a generated ID.
func NewThread() *Thread

// AddUser appends a user message.
func (t *Thread) AddUser(content string)

// Append adds the messages from a Run result and updates pending state.
func (t *Thread) Append(r *Result) {
    t.Messages = append(t.Messages, r.Messages...)
    t.PendingToolCalls = r.PendingToolCalls
}

// ResolvePending executes pending tool calls via the provided function,
// appends the results as RoleTool messages, and clears the pending state.
// Use after approval in a pause/resume workflow.
func (t *Thread) ResolvePending(ctx context.Context, fn func(ctx context.Context, call ToolCall) (*ToolResult, error)) error {
    for _, call := range t.PendingToolCalls {
        result, err := fn(ctx, call)
        if err != nil {
            return err
        }
        t.Messages = append(t.Messages, Message{
            Role:       RoleTool,
            ToolResult: result,
        })
    }
    t.PendingToolCalls = nil
    return nil
}
```

### Typical lifecycle

```go
// 1. Start a new thread (or load from storage)
thread := harness.NewThread()
thread.AddUser("What's the weather in Melbourne?")

// 2. Run the loop
result, err := harness.Run(ctx, provider,
    harness.WithMessages(thread.Messages...),
    harness.WithTools(tools...),
    harness.WithModel("claude-sonnet-4-20250514"),
    harness.WithMaxSteps(10),
)
thread.Append(result)
saveThread(thread) // persist to your storage

// 3. If paused for approval, resolve and continue
if result.StopReason == harness.StopPaused {
    // ... wait for user approval ...
    thread.ResolvePending(ctx, func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
        return executeTool(ctx, call) // your tool execution logic
    })
    saveThread(thread)

    // Continue the loop — the model sees the tool results and carries on
    result, err = harness.Run(ctx, provider,
        harness.WithMessages(thread.Messages...),
        harness.WithTools(tools...),
        harness.WithModel("claude-sonnet-4-20250514"),
        harness.WithMaxSteps(10),
    )
    thread.Append(result)
    saveThread(thread)
}

// 4. Next user message — just append and run again
thread.AddUser("How about Sydney?")
result, err = harness.Run(ctx, provider,
    harness.WithMessages(thread.Messages...),
    // ... same options
)
thread.Append(result)
saveThread(thread)
```

Persistence is batch — save after `Run()` returns, not during. If the process
dies mid-run, you re-run from the last persisted state. This keeps storage
integration simple and avoids partial-write issues.

## How Pincer Would Use This

```go
// Build tools from Pincer's existing tool implementations
tools := []harness.Tool{
    {
        ToolDef: harness.ToolDef{
            Name:        "web_search",
            Description: "Search the web using Kagi",
            Parameters:  webSearchSchema,
        },
        Execute: func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
            var args webSearchArgs
            json.Unmarshal(call.Arguments, &args)
            result, err := kagi.Search(ctx, args.Query)
            return &harness.ToolResult{Content: result}, err
        },
    },
    // ... other tools
}

// Load or create thread
thread := loadThread(threadID) // deserialize from SQLite
thread.AddUser(userMessage)

// Common options
runOpts := []harness.Option{
    harness.WithSystem(buildSystemPrompt()),
    harness.WithMessages(thread.Messages...),
    harness.WithTools(tools...),
    harness.WithModel("anthropic/claude-sonnet-4-20250514"),
    harness.WithMaxSteps(10),
    // The approval gate is a simple beforeTool hook
    harness.WithBeforeTool(func(ctx context.Context, call harness.ToolCall) (harness.ToolAction, error) {
        risk := classifyRisk(call.Name, call.Arguments)
        if risk == RiskREAD {
            return harness.ToolActionContinue, nil
        }
        return harness.ToolActionPause, nil
    }),
    harness.WithEventHandler(func(e harness.Event) {
        emitThreadEvent(threadID, e) // SSE to iOS client
    }),
}

result, err := harness.Run(ctx, provider, runOpts...)
thread.Append(result)
saveThread(thread) // persist to SQLite

// Handle pause/resume for approval
if result.StopReason == harness.StopPaused {
    // ... iOS shows approval UI, user approves ...
    thread.ResolvePending(ctx, executeApprovedTool)
    saveThread(thread)

    result, err = harness.Run(ctx, provider, runOpts...)
    thread.Append(result)
    saveThread(thread)
}
```

## What's Explicitly Out of Scope (v1)

- **Storage backends** — the `Thread` type serializes to JSON; the caller owns persistence (SQLite, files, etc.)
- **Routing/multi-agent** — Picoclaw's bus/routing is interesting but too opinionated
- **Memory/summarisation** — the caller handles context window management
- **Subagent spawning** — compose by calling `Run()` recursively if needed
- **System prompt construction** — the caller builds the prompt string
- **Steering/follow-up queues** — Pi's pattern is elegant but adds complexity; the `beforeTool` hook + `StopPaused` + resuming with accumulated messages covers the same use cases more simply

## Provider Packages

Thin adapters wrapping official SDKs. Each implements `harness.Provider` and
handles type mapping, streaming assembly, and provider-specific concerns
(prompt caching, thinking block round-trips) internally.

```
provider/
├── openai/        # wraps github.com/openai/openai-go
│   ├── provider.go
│   └── convert.go
└── anthropic/     # wraps github.com/anthropics/anthropic-sdk-go
    ├── provider.go
    └── convert.go
```

### Common pattern

Both adapters follow the same structure:

1. **Constructor** accepts provider-specific config (API key, base URL, default model)
2. **`Chat()`** maps harness types → SDK types, calls the SDK, maps response back
3. **Streaming** uses the SDK's built-in accumulator; calls `params.OnDelta()` per chunk
4. **Multi-turn** uses the SDK's `.ToParam()` to convert responses back to request format

### `provider/openai`

Wraps [`github.com/openai/openai-go`](https://github.com/openai/openai-go).
Covers OpenAI, OpenRouter, Groq, and any OpenAI-compatible endpoint.

```go
package openai

import (
    "github.com/openai/openai-go"
    harness "github.com/lox/agent-harness"
)

type Provider struct {
    client *openai.Client
    model  string
}

// Constructor — provider-specific config is typed, not map[string]any
func New(opts ...Option) *Provider

type Option func(*Provider)

func WithAPIKey(key string) Option
func WithBaseURL(url string) Option       // for OpenRouter, Groq, local models
func WithDefaultModel(model string) Option
```

**Type mapping:**

| Harness | openai-go |
|---|---|
| `Message{Role: RoleSystem}` | `openai.SystemMessage(content)` |
| `Message{Role: RoleUser}` | `openai.UserMessage(content)` |
| `Message{Role: RoleAssistant}` | `openai.ChatCompletionAssistantMessageParam{Content, ToolCalls}` |
| `Message{Role: RoleTool}` | `openai.ToolMessage(content, toolCallID)` |
| `ToolDef` | `openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{...})` |
| `ChatParams.Options["temperature"]` | `params.Temperature` |
| `ChatParams.Options["max_tokens"]` | `params.MaxCompletionTokens` |
| `Delta` | `ChatCompletionChunkChoiceDelta` fields |
| `Usage` | `CompletionUsage.PromptTokens / .CompletionTokens` |

**Streaming implementation:**

```go
func (p *Provider) Chat(ctx context.Context, params harness.ChatParams) (*harness.ChatResult, error) {
    reqParams := p.buildRequest(params)

    if params.OnDelta == nil {
        // Non-streaming: single request
        completion, err := p.client.Chat.Completions.New(ctx, reqParams)
        if err != nil {
            return nil, err
        }
        return p.convertResponse(completion), nil
    }

    // Streaming: use SDK accumulator
    stream := p.client.Chat.Completions.NewStreaming(ctx, reqParams)
    acc := openai.ChatCompletionAccumulator{}

    for stream.Next() {
        chunk := stream.Current()
        acc.AddChunk(chunk)

        if len(chunk.Choices) > 0 {
            delta := chunk.Choices[0].Delta
            params.OnDelta(harness.Delta{
                Text: delta.Content,
                ToolCall: convertToolCallDelta(delta.ToolCalls),
            })
        }
    }
    if err := stream.Err(); err != nil {
        return nil, err
    }

    return p.convertResponse(&acc.ChatCompletion), nil
}
```

**Provider-specific options** passed via `ChatParams.Options`:

```go
// Mapped from ChatParams.Options to openai.ChatCompletionNewParams
"temperature"       → params.Temperature
"max_tokens"        → params.MaxCompletionTokens
"top_p"             → params.TopP
"reasoning_effort"  → params.ReasoningEffort  // "low","medium","high"
"response_format"   → params.ResponseFormat   // "json_object", "text"
```

Unknown keys in `Options` are ignored with a warning log.

### `provider/anthropic`

Wraps [`github.com/anthropics/anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go).

```go
package anthropic

import (
    "github.com/anthropics/anthropic-sdk-go"
    harness "github.com/lox/agent-harness"
)

type Provider struct {
    client     *anthropic.Client
    model      string
    signatures map[int]string // thinking block signatures for round-trip
}

func New(opts ...Option) *Provider

type Option func(*Provider)

func WithAPIKey(key string) Option
func WithDefaultModel(model string) Option
func WithPromptCaching(enabled bool) Option // attach CacheControl to system/tools
```

**Type mapping:**

| Harness | anthropic-sdk-go |
|---|---|
| `WithSystem(str)` | `MessageNewParams.System: []TextBlockParam{{Text: str}}` |
| `Message{Role: RoleUser}` | `anthropic.NewUserMessage(anthropic.NewTextBlock(content))` |
| `Message{Role: RoleAssistant}` | `anthropic.NewAssistantMessage(blocks...)` — text + tool_use + thinking blocks |
| `Message{Role: RoleTool}` | `anthropic.NewUserMessage(anthropic.NewToolResultBlock(id, content, isErr))` |
| `ToolDef` | `anthropic.ToolUnionParam{OfTool: &ToolParam{Name, Description, InputSchema}}` |
| `Delta.Text` | `TextDelta.Text` from `ContentBlockDeltaEvent` |
| `Delta.Thinking` | `ThinkingDelta.Thinking` from `ContentBlockDeltaEvent` |
| `Delta.ToolCall` | `InputJSONDelta.PartialJSON` from `ContentBlockDeltaEvent` |
| `Usage` | `anthropic.Usage.InputTokens / .OutputTokens` |

**Key differences from the OpenAI adapter:**

1. **System prompt is separate** — not a message, it's the `System` field on `MessageNewParams`
2. **Tool results are user messages** — Anthropic uses `role=user` with `ToolResultBlock`
   content, not a separate `tool` role. The adapter handles this mapping.
3. **Thinking block round-trip** — Anthropic requires the cryptographic `Signature`
   from thinking blocks to be sent back in subsequent turns. The adapter tracks
   signatures internally (keyed by message index) and re-attaches them when
   converting harness messages back to Anthropic params. The harness `Message.Thinking`
   field stays a plain string.
4. **Prompt caching** — when `WithPromptCaching(true)` is set, the adapter attaches
   `CacheControlEphemeralParam` to the system prompt and tool definitions automatically.
   No harness-level changes needed.

**Streaming implementation:**

```go
func (p *Provider) Chat(ctx context.Context, params harness.ChatParams) (*harness.ChatResult, error) {
    reqParams := p.buildRequest(params)

    if params.OnDelta == nil {
        msg, err := p.client.Messages.New(ctx, reqParams)
        if err != nil {
            return nil, err
        }
        return p.convertResponse(msg), nil
    }

    stream := p.client.Messages.NewStreaming(ctx, reqParams)
    var acc anthropic.Message
    var currentBlockIndex int

    for stream.Next() {
        event := stream.Current()
        acc.Accumulate(event)

        switch ev := event.AsAny().(type) {
        case anthropic.ContentBlockStartEvent:
            currentBlockIndex = int(ev.Index)
        case anthropic.ContentBlockDeltaEvent:
            switch d := ev.Delta.AsAny().(type) {
            case anthropic.TextDelta:
                params.OnDelta(harness.Delta{Text: d.Text})
            case anthropic.ThinkingDelta:
                params.OnDelta(harness.Delta{Thinking: d.Thinking})
            case anthropic.InputJSONDelta:
                params.OnDelta(harness.Delta{
                    ToolCall: &harness.ToolCallDelta{
                        Index:     currentBlockIndex,
                        Arguments: d.PartialJSON,
                    },
                })
            }
        }
    }
    if err := stream.Err(); err != nil {
        return nil, err
    }

    // Stash thinking signatures for round-trip
    p.stashSignatures(&acc)

    return p.convertResponse(&acc), nil
}
```

**Provider-specific options** passed via `ChatParams.Options`:

```go
"temperature"       → params.Temperature
"max_tokens"        → params.MaxTokens      // required by Anthropic
"top_p"             → params.TopP
"top_k"             → params.TopK
"thinking_budget"   → params.Thinking       // ThinkingConfigParamOfEnabled(budget)
```

### Usage with the harness

```go
// OpenAI / OpenRouter
provider := openai.New(
    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    openai.WithDefaultModel("gpt-4o"),
)

// Anthropic
provider := anthropic.New(
    anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
    anthropic.WithDefaultModel("claude-sonnet-4-20250514"),
    anthropic.WithPromptCaching(true),
)

// Both implement harness.Provider — use identically
result, err := harness.Run(ctx, provider,
    harness.WithMessages(thread.Messages...),
    harness.WithTools(tools...),
    harness.WithMaxSteps(10),
)
```

### Testing

Both adapters should be testable without real API calls. The official SDKs
accept `option.WithHTTPClient()` / `option.WithBaseURL()`, so tests can
point at a local HTTP server that returns canned responses:

```go
func TestOpenAIProvider(t *testing.T) {
    server := httptest.NewServer(mockOpenAIHandler())
    defer server.Close()

    provider := openai.New(
        openai.WithBaseURL(server.URL),
        openai.WithAPIKey("test-key"),
    )
    result, err := provider.Chat(ctx, harness.ChatParams{...})
    // assert result
}
```

## Design Guidance

Patterns from production agent systems that callers should be aware of.

### Progressive disclosure

The single highest-impact harness pattern. Agents perform better with fewer,
more relevant tools than with everything available at once. Cursor saw 46.9%
token reduction from lazy tool loading. Vercel went from failing to passing
by removing 80% of tools.

Use `WithToolFilter` to expose only relevant tools per step. A common
pattern is starting with read-only tools and adding write tools only after
the model has gathered context:

```go
harness.WithToolFilter(func(step int, messages []Message) []Tool {
    if step == 0 {
        return readOnlyTools
    }
    return allTools
})
```

### KV-cache stability vs progressive disclosure

These two patterns are in tension. Manus keeps all tools permanently loaded
because changing tool definitions invalidates the LLM's KV cache for all
subsequent tokens. Cursor loads tools lazily and accepts the cache cost.

Rule of thumb: if your agent typically runs <5 steps, progressive disclosure
wins (fewer tokens per call matters more). If it runs 20+ steps with stable
tool sets, cache stability wins (amortised prefix caching matters more).

When not using `WithToolFilter`, tools are sorted by name for stable cache
keys across calls.

### Tool result reminders

Claude Code injects fixed text after every tool execution ("system
reminders") for higher behavioural adherence than system-prompt-only
instructions. This works because it places the reminder in the model's
recent attention window on every turn.

Implement this in your tool's `Execute` function:

```go
Execute: func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
    result := doWork(call)
    result.Content += "\n\nReminder: always run tests after editing code."
    return result, nil
}
```

Or centralise it via `WithAfterTool` if you want the same suffix on every
tool.

### Keep the harness simple

Every team in the research (Anthropic, Manus, Cursor) converges on
simplifying their harness over time. Manus rewrote theirs five times, each
time removing things. Anthropic designs Claude Code's scaffold to shrink as
models improve.

If your harness is getting more complex while models get better, something
is wrong. Resist the urge to add orchestration layers, competing agent
personas, or DAG-based control flow. The single flat loop works.

## Future Considerations (v2+)

- **Parallel tool execution** — opt-in via an option; execute independent tools concurrently
- **Fallback chain** — Picoclaw's error classification + ordered fallback pattern, as a `Provider` wrapper
- **Context compression middleware** — a `Provider` wrapper that monitors token usage and auto-summarises when approaching limits
- **MCP client** — tool discovery and execution via Model Context Protocol

## References

### Agent harness architecture
- Anthropic — [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Sep 2025)
- Anthropic — [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) (Jan 2026)
- LangChain — [Deep Agents](https://blog.langchain.com/deep-agents/) (Jul 2025)
- LangChain — [Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) (Feb 2026)
- Horthy — [12 Factor Agents](https://paddo.dev/blog/12-factor-agents/)

### Production agent case studies
- Cursor — [Dynamic Context Discovery](https://cursor.com/blog/dynamic-context-discovery) (Jan 2026)
- Cursor — [Improving Agent with Semantic Search](https://cursor.com/blog/semsearch) (2026)
- Cursor — [Improving Cursor's Agent for OpenAI Codex Models](https://cursor.com/blog/codex-model-harness) (2026)
- Manus — [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) (Jul 2025)
- Cognition — [Devin's 2025 Performance Review](https://cognition.ai/blog/devin-annual-performance-review-2025) (2025)
- Phil Schmid — [Context Engineering for AI Agents: Part 2](https://www.philschmid.de/context-engineering-part-2) (Dec 2025)

### Claude Code reverse engineering
- PromptLayer — [Claude Code: Behind-the-Scenes of the Master Agent Loop](https://blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop/)
- Vrungta — [Claude Code Architecture (Reverse Engineered)](https://vrungta.substack.com/p/claude-code-architecture-reverse)
- Jannesklaas — [Agent Design Lessons from Claude Code](https://jannesklaas.github.io/ai/2025/07/20/claude-code-agent-design.html)

### Research
- Yang et al. — [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793) (NeurIPS 2024)
- Liu et al. — [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (TACL 2024)

### Progressive disclosure
- Honra — [Why AI Agents Need Progressive Disclosure, Not More Data](https://www.honra.io/articles/progressive-disclosure-for-ai-agents)
- Karpathy — [2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/)

### Repositories studied for this design
- [Pi agent package](https://github.com/badlogic/pi-mono/tree/main/packages/agent) — TypeScript agent loop with steering/follow-up queues and EventStream abstraction
- [Picoclaw](https://github.com/sipeed/picoclaw) — Go agent framework with minimal LLMProvider interface and extracted RunToolLoop
- [Pincer](https://github.com/lox/pincer) — Go security-first agent (target consumer for this library)
