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
5. Response conversion maps usage, tool calls, response identity, finish state,
   and opaque round-trip data back to harness types.

## Result Contract

Adapters should populate `ChatResult.ResponseID`, `FinishReason`,
`FinishDetails`, and `Usage` whenever the provider exposes them. Finish reasons
must use the provider-neutral harness values rather than passing through raw
provider strings:

- natural completion -> `FinishReasonEndTurn`
- complete local tool calls -> `FinishReasonToolUse`
- refusal -> `FinishReasonRefusal`
- output-token exhaustion -> `FinishReasonMaxTokens`
- other partial output -> `FinishReasonIncomplete`
- provider-requested resubmission -> `FinishReasonContinuation`

If a response contains provider-native state needed on a later request, store
its serialized representation in `Message.ProviderData` and restore it when
converting that message back to the provider SDK. The core deliberately treats
this field as opaque JSON. This keeps signed thinking, redacted content, server
tool blocks, and continuation payloads intact without making them part of the
harness model.

`Usage.InputTokens` contains uncached input when the provider reports uncached
and cached input separately; otherwise it contains the provider's primary input
count. Populate cache fields independently from the provider response. The
5-minute and 1-hour creation counters are subdivisions of
`CacheCreationInputTokens`, and `CacheReadInputTokens` may be the source for the
normalized `CachedInputTokens` value when a provider uses cache-read terminology.

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

Target SDK: `github.com/openai/openai-go`

Status: implemented with non-streaming and streaming support.

Constructor options:

- `WithAPIKey(key string)`
- `WithBaseURL(url string)`
- `WithDefaultModel(model string)`
- `WithRequestOption(opt)`

`ChatParams.Options` mappings:

- `temperature` -> request temperature field
- `max_tokens` -> completion token limit
- `top_p` -> nucleus sampling field
- `reasoning_effort` -> model reasoning effort where supported
- `response_format` -> JSON/text response mode

Unknown option keys should be ignored safely (without failing the run).

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

## Testing Approach

- Use `httptest.NewServer` to return canned provider responses
- Verify message and tool-call mapping in both directions
- Verify streaming delta conversion and final accumulation
- Verify usage mapping and error handling semantics

## Design Guidance For Callers

- Prefer adapter constructor options for static provider config and reserve `ChatParams.Options` for per-request tuning.
- Keep one harness call path for all providers; provider differences should stay inside adapters.
- Add integration tests per adapter before enabling by default in examples or docs.
