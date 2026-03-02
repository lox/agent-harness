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

## Status

- Core harness loop, hooks, and thread state are implemented
- Unit tests are in place for core loop behaviour and pause/resume
- OpenAI provider adapter is implemented (`provider/openai`)
- Anthropic provider adapter is implemented (`provider/anthropic`)
- `examples/claw` provides a REPL harness for manual testing

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

## Roadmap

- [x] Extract the reusable harness core (`Run`, messages, tools, provider interface)
- [x] Add pause/resume support (`StopPaused`, `PendingToolCalls`, `Thread.ResolvePending`)
- [x] Add lifecycle hooks and event emission
- [x] Stabilise core loop semantics with unit tests
- [x] Add a runnable REPL example under `examples/claw`
- [x] Add CI for `go test`, `go test -race`, and `go vet`
- [x] Implement `provider/openai` adapter (non-streaming + streaming)
- [x] Implement `provider/anthropic` adapter (non-streaming + streaming)
- [x] Add provider integration tests using local HTTP test servers
- [ ] Cut `v0.1.0` once adapters + example + CI are complete

## Documentation

- [docs/architecture.md](docs/architecture.md) — API shape, loop lifecycle, and state model
- [docs/runner.md](docs/runner.md) — optional helper for starting/stopping active runs
- [docs/providers.md](docs/providers.md) — provider adapter contracts and type mappings
- [docs/research.md](docs/research.md) — research notes and design rationale

The `docs/` directory is the source of truth for design and implementation guidance.

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

## Cancelling Active Runs

Use `runner.Runner` when you want to interrupt an in-flight run from external control input such as a user saying "stop".

```go
r := runner.New()

done, err := r.Start(context.Background(), thread.ID, func(ctx context.Context) error {
    result, err := harness.Run(ctx, provider,
        harness.WithMessages(thread.Messages...),
        harness.WithTools(tools...),
    )
    if err == nil {
        thread.Append(result)
    }
    return err
})
if err != nil {
    return err
}

// elsewhere: control-plane stop command
if strings.EqualFold(strings.TrimSpace(userInput), "stop") {
    r.Stop(thread.ID)
}

runErr := <-done
_ = runErr
```

## Running Claw Example

```bash
OPENAI_API_KEY=... go run ./examples/claw
```

Then type prompts or control commands (`/stop`, `/history`, `/tools`, `/quit`).

See [docs/architecture.md](docs/architecture.md) for the primary implementation guide.
