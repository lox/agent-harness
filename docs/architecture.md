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
├── option.go
└── memory/
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
- `ChatParams` includes model, system prompt, history, tool definitions,
  provider-neutral reasoning controls, provider options, and an optional
  streaming callback
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

`Usage` splits tokens into priceable categories. `InputTokens` is ordinary,
uncached input; `CachedInputTokens` is the normalized cache-read count;
`CacheCreationInputTokens` is the aggregate cache-write count; and
`OutputTokens` is the provider's inclusive output total. OpenAI's
`cache_write_tokens` maps to cache creation, while Anthropic's 5-minute and
1-hour creation counts remain subdivisions of the aggregate rather than extra
tokens to add to it. `CacheReadInputTokens` preserves provider cache-read
terminology and normally equals `CachedInputTokens`, so those two fields must
not be summed.

`Result.CallUsage` contains one `Usage` value for each completed provider call,
in step order, and `Result.TotalUsage` is their field-wise sum. Price each
`CallUsage` entry independently when a provider applies thresholds or
multipliers per response, then add the resulting costs. A zero entry means the
provider omitted usage for that step.

`ReasoningOptions` provides shared `Effort` and `Mode` fields through
`WithReasoning`. Providers map the fields they support. These generic values
take precedence over compatibility keys in `WithProviderOptions`.

`WithPreviousResponseID` resumes provider-owned response state on the first
call of a run. Later calls leave that option empty and continue through the
response IDs stored in newly appended assistant messages, so an external ID is
never replayed over a newer continuation.

### Tool Model

- `ToolDef` carries schema for the model
- `Tool` bundles schema + `Execute` function
- `ToolResult` separates LLM-facing content from caller metadata and optional user-facing content

## Run Lifecycle

`Run` executes a bounded turn loop.

1. Validate options and seed message history
2. Select active tools (optionally via `WithToolFilter`)
3. Call provider with `ChatParams`
4. Retain that call's usage, add it to total usage, append the assistant
   message, and emit events
5. Interpret the provider finish reason
6. On `end_turn`, `refusal`, or incomplete output, return the matching stop state
7. On `continuation`, append the assistant message unchanged and call the provider again
8. On `tool_use`, when another provider turn remains, apply `WithBeforeTool`, execute each call, and append its `RoleTool` result
9. Apply `WithAfterTool`
10. Repeat until a terminal state, hook pause, cancellation, or max-steps

Once a provider response has been recorded, `Run` returns the partial `Result`
alongside any later error. Cancelled calls and calls abandoned by a failing hook
receive error tool results so the transcript remains valid. Callers that persist
threads should append a non-nil result before handling the error.

## Stop Reasons

- `StopEndTurn`: assistant returned no tool calls
- `StopMaxSteps`: configured step limit reached
- `StopCancelled`: context cancelled or deadline exceeded
- `StopPaused`: hook requested pause; pending calls returned in `Result.PendingToolCalls`
- `StopRefusal`: provider refused the request; details remain in `Result.FinishDetails`
- `StopIncomplete`: provider returned `max_tokens` or another incomplete state;
  the exact state remains in `Result.FinishReason`
- `StopError`: the run returned an error after producing a partial, persistable
  result

Provider continuation is not a stop reason. It consumes a step and continues
inside `Run`, so repeated continuation responses eventually return
`StopMaxSteps`. `Result.ResponseID` retains the most recent non-empty provider
response ID and `Result.FinishReason` retains the last observed provider state.
If the last permitted provider turn returns `tool_use`, the assistant message
and its metadata remain in the result, but the tools are not executed because
their results could not be sent to another provider turn.

## Thread State Model

`Thread` is a thin persistence-ready state container.

- `AddUser(content)` appends user input
- `Append(result)` appends run outputs and updates pending calls
- `ResolvePending(ctx, fn)` commits each successful call immediately, so a later
  failure leaves only the unexecuted calls pending

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
- before-tool hooks must return a defined `ToolAction`; unknown values fail closed

## Optional Memory Package

The core harness remains storage-agnostic. The optional `memory` package adds a
file-backed memory layer that applications can compose into their own prompts
and tools.

- `MEMORY.md` is curated long-term memory.
- `memory/*.md` contains dated working memory and session captures.
- `Store.Bootstrap` appends the memory prompt section to the system prompt.
- `Store.Tools` exposes `memory_search` and `memory_get`.
- `CaptureThread` and `CaptureMessages` support `/new`, `/reset`, and
  pre-compaction flush flows.
- `PromotionCandidates`, `ApplyPromotions`, and `NewConsolidator` provide a
  reviewable consolidation path from recalled working memory into `MEMORY.md`.

See [memory.md](memory.md) for usage details.
