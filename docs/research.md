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

## Design Documentation

This research explains why we made the harness choices; the implemented design is documented in:

- [docs/architecture.md](architecture.md) for API shape, lifecycle, events, and validation rules
- [README.md](../README.md) for public usage, current status, and examples

## Key Learnings Applied To The Design

- Harness structure matters as much as model choice; published eval deltas from scaffold changes justify keeping the core loop as a first-class product concern.
- The winning baseline is a flat, bounded tool loop (`Run`) with clear stop reasons, rather than layered orchestration by default.
- Progressive disclosure is an architectural pattern, not a prompt tweak; `WithToolFilter` is intentionally first-class.
- Separating model-facing and user-facing tool output improves reliability and operator UX (`ToolResult.Content` vs `UserContent` and metadata).
- Pause/resume must be explicit and serialisable for approvals; `StopPaused` + `PendingToolCalls` + `Thread.ResolvePending` keeps this state machine simple.
- The harness stays stateless and storage-agnostic so apps can compose their own persistence and policy layers cleanly.

## Operational Guidance From Research

- Start with minimal tool sets and expose additional tools only when the model has gathered enough context.
- Keep tool definitions stable within a run where possible, because prefix churn can hurt cache effectiveness on long trajectories.
- Prefer primitives over broad integrations; small, predictable tools outperform large registries in most production reports.
- Keep scaffold complexity trending down as models improve; avoid adding orchestration layers unless evaluation data proves a net benefit.

## References

### Agent harness architecture

- [Anthropic — Effective Context Engineering for AI Agents (Sep 2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Anthropic — Effective Harnesses for Long-Running Agents (Jan 2026)](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [LangChain — Deep Agents (Jul 2025)](https://blog.langchain.com/deep-agents/)
- [LangChain — Improving Deep Agents with Harness Engineering (Feb 2026)](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)
- [Horthy — 12 Factor Agents](https://paddo.dev/blog/12-factor-agents/)

### Production case studies

- [Cursor — Dynamic Context Discovery (Jan 2026)](https://cursor.com/blog/dynamic-context-discovery)
- [Cursor — Improving Agent with Semantic Search (2026)](https://cursor.com/blog/semsearch)
- [Cursor — Improving Cursor's Agent for OpenAI Codex Models (2026)](https://cursor.com/blog/codex-model-harness)
- [Manus — Context Engineering for AI Agents: Lessons from Building Manus (Jul 2025)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [Cognition — Devin's 2025 Performance Review](https://cognition.ai/blog/devin-annual-performance-review-2025)

### Additional research

- [Yang et al. — SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (NeurIPS 2024)](https://arxiv.org/abs/2405.15793)
- [Liu et al. — Lost in the Middle: How Language Models Use Long Contexts (TACL 2024)](https://arxiv.org/abs/2307.03172)

### Secondary analyses and commentary

- [PromptLayer — Claude Code: Behind-the-Scenes of the Master Agent Loop](https://blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop/)
- [Vrungta — Claude Code Architecture (Reverse Engineered)](https://vrungta.substack.com/p/claude-code-architecture-reverse)
- [Honra — Why AI Agents Need Progressive Disclosure, Not More Data](https://www.honra.io/articles/progressive-disclosure-for-ai-agents)
- [Phil Schmid — Context Engineering for AI Agents: Part 2 (Dec 2025)](https://www.philschmid.de/context-engineering-part-2)
- [Karpathy — 2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/)
- [Jannes Klaas — Agent Design Lessons from Claude Code](https://jannesklaas.github.io/ai/2025/07/20/claude-code-agent-design.html)
