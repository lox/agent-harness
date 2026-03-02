# Provider Adapters

This document defines the adapter shape and expected behaviour for provider packages.

## Adapter Principles

- Adapters implement `harness.Provider`
- Keep adapter code thin: map harness types to SDK types and back
- Keep provider-specific settings in adapter options
- Support both non-streaming and streaming paths
- Keep adapter tests network-free via local HTTP servers

## Common Adapter Pattern

All adapters should follow the same flow:

1. Constructor accepts typed provider config (`api key`, `default model`, `base URL`, provider flags).
2. `Chat()` maps harness request types to SDK request types.
3. Non-streaming path issues one request and converts final response.
4. Streaming path emits `Delta` chunks via `ChatParams.OnDelta` and accumulates to one final message.
5. Response conversion maps usage and tool calls back to harness types.

## Planned Packages

```
provider/
├── openai/
└── anthropic/
```

Current status:

- `provider/openai`: implemented
- `provider/anthropic`: planned

## OpenAI Adapter

Target SDK: `github.com/openai/openai-go`

Status: implemented with non-streaming and streaming support.

Constructor options:

- `WithAPIKey(key string)`
- `WithBaseURL(url string)`
- `WithDefaultModel(model string)`

`ChatParams.Options` mappings:

- `temperature` -> request temperature field
- `max_tokens` -> completion token limit
- `top_p` -> nucleus sampling field
- `reasoning_effort` -> model reasoning effort where supported
- `response_format` -> JSON/text response mode

Unknown option keys should be ignored safely (without failing the run).

## Anthropic Adapter

Target SDK: `github.com/anthropics/anthropic-sdk-go`

Status: planned.

Constructor options:

- `WithAPIKey(key string)`
- `WithDefaultModel(model string)`
- `WithPromptCaching(enabled bool)`

`ChatParams.Options` mappings:

- `temperature`
- `max_tokens`
- `top_p`
- `top_k`
- `thinking_budget`

Adapter-specific mapping considerations:

- System prompt is a top-level request field, not a normal message.
- Tool result responses are represented as user-role tool-result blocks by the SDK.
- Thinking blocks may require signature round-tripping between turns.
- Prompt caching can be adapter-managed without harness API changes.

## Testing Approach

- Use `httptest.NewServer` to return canned provider responses
- Verify message and tool-call mapping in both directions
- Verify streaming delta conversion and final accumulation
- Verify usage mapping and error handling semantics

## Design Guidance For Callers

- Prefer adapter constructor options for static provider config and reserve `ChatParams.Options` for per-request tuning.
- Keep one harness call path for all providers; provider differences should stay inside adapters.
- Add integration tests per adapter before enabling by default in examples or docs.
