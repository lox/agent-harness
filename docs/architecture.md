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

### Provider Contract

- `Provider` exposes one method: `Chat(context.Context, ChatParams) (*ChatResult, error)`
- `ChatParams` includes model, system prompt, history, tool definitions, provider options, and optional streaming callback

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
5. If no tool calls: stop with `StopEndTurn`
6. For each tool call:
7. Apply `WithBeforeTool` action (`Continue`, `Skip`, `Pause`)
8. Execute tool and append `RoleTool` result message
9. Apply `WithAfterTool`
10. Repeat until natural stop, pause, cancellation, or max-steps

## Stop Reasons

- `StopEndTurn`: assistant returned no tool calls
- `StopMaxSteps`: configured step limit reached
- `StopCancelled`: context cancelled or deadline exceeded
- `StopPaused`: hook requested pause; pending calls returned in `Result.PendingToolCalls`

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
