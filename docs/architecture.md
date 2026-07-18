# Architecture

This document describes the implemented harness API and lifecycle.

## Design Goals

- Keep the API small and composable
- Decouple loop mechanics from application concerns
- Support provider/tool pluggability via interfaces and functions
- Make pause/resume explicit for approval workflows

## Package Layout

```
agent-harness/
├── provider.go
├── tool.go
├── message.go
├── thread.go
├── loop.go
├── event.go
└── option.go
```

## Core Types

### Message Model

- `Message` is the canonical conversation unit
- `ToolCall` is requested by the model in assistant messages
- Tool outputs are appended as separate `RoleTool` messages
- `Message.ProviderData` carries opaque adapter-owned JSON that must survive a
  provider round trip, such as signed reasoning or native continuation blocks

### Provider Contract

- `Provider` exposes one method: `Chat(context.Context, ChatParams) (*ChatResult, error)`
- `ChatParams` includes model, system prompt, history, tool definitions, provider options, and optional streaming callback
- `ChatResult` includes the assistant message, provider response ID, finish
  reason and details, and optional usage

`FinishReason` is the provider-neutral response state:

- `FinishReasonEndTurn`: natural completion without tool calls
- `FinishReasonToolUse`: a complete set of tool calls is ready to execute
- `FinishReasonRefusal`: the provider refused the request
- `FinishReasonMaxTokens`: output stopped at the configured token limit
- `FinishReasonIncomplete`: output is incomplete for another reason
- `FinishReasonContinuation`: the provider asks the harness to submit the
  response again before treating it as terminal, such as an Anthropic
  `pause_turn`

Providers should always set a finish reason. For compatibility with providers
written before this contract, an unspecified reason is inferred from the
presence of tool calls. Unknown reasons and inconsistent states such as
`end_turn` with tool calls return an error, so explicit refusal or incomplete
output can never become `StopEndTurn` silently.

`Usage` keeps the provider's input and output counts alongside cached input,
cache creation, and cache read counts. The 5-minute and 1-hour creation fields
are details within the cache creation total rather than extra tokens to add to
it. Providers that expose both a normalized cached-input total and a cache-read
counter may populate both; callers should use the field appropriate to their
accounting model rather than sum every cache field together.

### Tool Model

- `ToolDef` carries schema for the model
- `Tool` bundles schema + `Execute` function
- `ToolResult` separates LLM-facing content from caller metadata and optional user-facing content

## Run Lifecycle

`Run` executes a bounded turn loop.

1. Validate options and seed message history
2. Select active tools (optionally via `WithToolFilter`)
3. Call provider with `ChatParams`
4. Append assistant message and emit events
5. Interpret the provider finish reason
6. On `end_turn`, `refusal`, or incomplete output, return the matching stop state
7. On `continuation`, append the assistant message unchanged and call the provider again
8. On `tool_use`, apply `WithBeforeTool`, execute each call, and append its `RoleTool` result
9. Apply `WithAfterTool`
10. Repeat until a terminal state, hook pause, cancellation, or max-steps

## Stop Reasons

- `StopEndTurn`: assistant returned no tool calls
- `StopMaxSteps`: configured step limit reached
- `StopCancelled`: context cancelled or deadline exceeded
- `StopPaused`: hook requested pause; pending calls returned in `Result.PendingToolCalls`
- `StopRefusal`: provider refused the request; details remain in `Result.FinishDetails`
- `StopIncomplete`: provider returned `max_tokens` or another incomplete state;
  the exact state remains in `Result.FinishReason`

Provider continuation is not a stop reason. It consumes a step and continues
inside `Run`, so repeated continuation responses eventually return
`StopMaxSteps`. `Result.ResponseID` retains the most recent non-empty provider
response ID and `Result.FinishReason` retains the last observed provider state.

## Thread State Model

`Thread` is a thin persistence-ready state container.

- `AddUser(content)` appends user input
- `Append(result)` appends run outputs and updates pending calls
- `ResolvePending(ctx, fn)` executes pending calls and appends tool result messages

## Event Model

The harness emits structured events through `WithEventHandler`.

- `EventTurnStart`
- `EventTurnEnd`
- `EventToolStart`
- `EventToolEnd`
- `EventMessage`
- `EventError`

Event payloads are emitted from stable value copies to avoid pointer mutation issues in asynchronous consumers.

## Validation Rules

- `maxSteps` must be `> 0`
- tools must have unique non-empty names
- tools must have a non-nil `Execute`
- filtered tool sets are revalidated per-step
