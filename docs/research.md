# Research Notes

This document captures the core research that informed the harness design.

## Motivation

Most agent applications repeat the same mechanics:

1. call model
2. parse tool calls
3. execute tools
4. append tool results
5. continue until completion

The harness extracts that reusable loop and leaves app-specific concerns (storage, approval UX, persistence, routing) to consumers.

## Systems Studied

## Pi (TypeScript)

- Async generator/event-stream loop model
- Distinct steering/follow-up queues for external control
- Strong separation of model-facing output vs UI-facing details

## Picoclaw (Go)

- Minimal provider interface (`Chat`)
- Reusable standalone tool loop
- Valuable `ForLLM`/`ForUser` split for tool outputs
- Stable tool ordering and fallback patterns

## Pincer (Go target consumer)

- Current planner loop mixes orchestration and app policy
- Tool registration/execution split across multiple places
- Approval flow is powerful but tightly coupled
- Step budgeting and continuation logic are scattered

## Key Design Choices

- Keep v1 to a single `Run` loop and compact interfaces
- Use callbacks/hooks over framework inheritance
- Keep pause/resume explicit using `StopPaused` + `PendingToolCalls`
- Keep harness stateless and storage-agnostic with `Thread` as optional convenience
- Treat progressive disclosure (`WithToolFilter`) as first-class

## Deliberate v1 Exclusions

- Built-in storage backends
- Multi-agent routing and orchestration layers
- Automatic context compression/summarisation
- Built-in provider fallback chains
- MCP integration baked into core loop

## References

### Agent harness architecture

- Anthropic — Effective Context Engineering for AI Agents (Sep 2025)
- Anthropic — Effective Harnesses for Long-Running Agents (Jan 2026)
- LangChain — Deep Agents (Jul 2025)
- LangChain — Improving Deep Agents with Harness Engineering (Feb 2026)
- Horthy — 12 Factor Agents

### Production case studies

- Cursor — Dynamic Context Discovery (Jan 2026)
- Cursor — Improving Agent with Semantic Search (2026)
- Cursor — Improving Cursor's Agent for OpenAI Codex Models (2026)
- Manus — Context Engineering for AI Agents (Jul 2025)
- Cognition — Devin's 2025 Performance Review

### Additional research

- SWE-agent (NeurIPS 2024)
- Lost in the Middle (TACL 2024)
