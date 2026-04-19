# Provider Adapters

This document defines the adapter shape and expected behaviour for provider packages.

## Adapter Principles

- Adapters implement `harness.Provider`
- Keep adapter code thin: map harness types to SDK types and back
- Normalize harness messages into a shared internal role-plus-part conversation model first
- Keep provider-specific settings in adapter options
- Support both non-streaming and streaming paths
- Keep adapter tests network-free via local HTTP servers

## Common Adapter Pattern

All adapters should follow the same flow:

1. Constructor accepts typed provider config (`api key`, `default model`, `base URL`, provider flags).
2. `Chat()` normalizes harness request types into provider-neutral conversation entries, then maps them to SDK request types.
3. Non-streaming path issues one request and converts final response.
4. Streaming path emits `Delta` chunks via `ChatParams.OnDelta` and accumulates to one final message.
5. Response conversion maps usage and tool calls back to harness types.

## Provider Packages

```
provider/
├── openai/
└── anthropic/
```

Current status:

- `provider/openai`: implemented
- `provider/anthropic`: implemented

## OpenAI Adapter

Target SDK: `github.com/openai/openai-go/v3`

Status: implemented with non-streaming and streaming support via the Responses API.

Constructor options:

- `WithAPIKey(key string)`
- `WithBaseURL(url string)`
- `WithDefaultModel(model string)`
- `WithRequestOption(opt)`

Transport notes:

- Requests go to `/responses`, not `/chat/completions`
- Custom base URLs must expose a Responses-compatible endpoint if you want GPT-5 reasoning plus tools to work correctly

`ChatParams.Options` mappings:

- `temperature` -> request temperature field
- `max_tokens` -> completion token limit
- `top_p` -> nucleus sampling field
- `reasoning_effort` -> raw OpenAI reasoning effort string where supported (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, subject to model support)
- `response_format` -> JSON/text response mode

Unknown option keys should be ignored safely (without failing the run).

Model IDs are also passed through as strings, so callers can use raw model IDs such as `gpt-5.4` or the v3 SDK constants interchangeably.
The repo is pinned to the v3 SDK line, which includes generated constants for `gpt-5.4` and `xhigh`.
The adapter keeps the public harness `ChatParams` surface unchanged while mapping thread history to Responses input items statelessly.

## Anthropic Adapter

Target SDK: `github.com/anthropics/anthropic-sdk-go`

Status: implemented with non-streaming and streaming support.

Constructor options:

- `WithAPIKey(key string)`
- `WithBaseURL(url string)`
- `WithDefaultModel(model string)`
- `WithRequestOption(opt)`

`ChatParams.Options` mappings:

- `temperature`
- `max_tokens`
- `top_p`
- `top_k`
- `thinking_budget`

Adapter-specific mapping considerations:

- System prompt is a top-level request field, not a normal message.
- Tool result responses are represented as user-role tool-result blocks by the SDK.
- Thinking content is mapped to `Message.Thinking` when present.
- Request conversion now starts from the same shared internal role-plus-part conversation model used by the OpenAI adapter.

## Testing Approach

- Use `httptest.NewServer` to return canned provider responses
- Verify message and tool-call mapping in both directions
- Verify streaming delta conversion and final accumulation
- Verify usage mapping and error handling semantics

## Design Guidance For Callers

- Prefer adapter constructor options for static provider config and reserve `ChatParams.Options` for per-request tuning.
- Keep one harness call path for all providers; provider differences should stay inside adapters.
- Add integration tests per adapter before enabling by default in examples or docs.
