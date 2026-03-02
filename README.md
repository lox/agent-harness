# 🧰 agent-harness

A minimal, composable Go library for building agentic tool-calling loops on top of LLM APIs.

## Requirements

- Go `1.26+`

```go
result, err := harness.Run(ctx, provider,
    harness.WithSystem("You are a helpful assistant."),
    harness.WithMessages(thread.Messages...),
    harness.WithTools(tools...),
    harness.WithModel("claude-opus-4-6"),
    harness.WithMaxSteps(10),
)
```

## What it does

Implements the core agent loop: call the LLM → execute tool calls → feed results back → repeat. Everything else (storage, prompts, routing) is your problem.

## What it doesn't do

- Bundle LLM provider clients — you implement a single `Chat()` method
- Manage conversation storage — you serialise the `Thread` type however you want
- Construct system prompts — you pass a string
- Orchestrate multi-agent workflows — call `Run()` from a tool for sub-agents

## Design

- Single `Run()` function, not a framework
- `Provider` interface with one method
- Tools bundle schema + execution in one place
- Hooks for approval gates (`WithBeforeTool`), streaming (`WithOnDelta`), and observability (`WithEventHandler`)
- Progressive disclosure via `WithToolFilter`
- Pause/resume with explicit `PendingToolCalls` for approval workflows
- Composes naturally with [ACP](https://agentclientprotocol.com/) and [MCP](https://modelcontextprotocol.io/)

## Pause And Resume

```go
thread := harness.NewThread()
thread.AddUser("Delete old preview deployments")

result, err := harness.Run(ctx, provider,
    harness.WithMessages(thread.Messages...),
    harness.WithTools(tools...),
    harness.WithBeforeTool(func(ctx context.Context, call harness.ToolCall) (harness.ToolAction, error) {
        if call.Name == "delete_deployment" {
            return harness.ToolActionPause, nil
        }
        return harness.ToolActionContinue, nil
    }),
)
if err != nil {
    return err
}

thread.Append(result)

if result.StopReason == harness.StopPaused {
    // approval flow happens outside the harness
    err = thread.ResolvePending(ctx, func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
        return executeApprovedTool(ctx, call)
    })
    if err != nil {
        return err
    }

    result, err = harness.Run(ctx, provider,
        harness.WithMessages(thread.Messages...),
        harness.WithTools(tools...),
    )
}
```

## Progressive Tool Disclosure

```go
result, err := harness.Run(ctx, provider,
    harness.WithMessages(thread.Messages...),
    harness.WithTools(readTool, writeTool),
    harness.WithToolFilter(func(step int, _ []harness.Message) []harness.Tool {
        if step == 0 {
            return []harness.Tool{readTool}
        }
        return []harness.Tool{readTool, writeTool}
    }),
)
```

See [PLAN.md](PLAN.md) for the full design, API types, pseudocode, and research notes.
