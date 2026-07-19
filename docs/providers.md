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

`Usage.InputTokens` contains ordinary, uncached input. When a provider reports
an inclusive input total, the adapter subtracts cache reads and cache writes.
`CachedInputTokens` is the normalized cache-read count and
`CacheCreationInputTokens` is the normalized cache-write count. Provider cache
detail remains available in `CacheReadInputTokens` and the 5-minute and 1-hour
creation counters. The TTL counters subdivide `CacheCreationInputTokens`, and
`CacheReadInputTokens` normally duplicates `CachedInputTokens`, so pricing code
must not sum either pair.

The loop copies each completed call's usage into `Result.CallUsage` before
adding it to `Result.TotalUsage`. The slice stays in provider-call order and has
the same length as `Result.Steps`; a zero entry records a call whose provider
omitted usage. This lets callers apply per-response pricing rules, including
long-context multipliers, before summing costs.

Use `harness.WithReasoning(harness.ReasoningOptions{...})` for shared reasoning
effort and mode controls. Adapters may keep string-keyed compatibility options,
but the generic fields take precedence when both are present.

Use `harness.WithPreviousResponseID(id)` to resume an existing provider response
at the start of a run. The core supplies it only to the first provider call;
subsequent stateful calls use response IDs captured during that run.

## Provider Packages

```
provider/
├── openai/
└── anthropic/
```

Current status:

- `provider/openai`: implemented Responses API adapter
- `provider/anthropic`: implemented

## OpenAI Responses Adapter

Target SDK: `github.com/openai/openai-go/v3`

Package: `provider/openai`

The OpenAI adapter uses the Responses API. It supports full input history and
stores response IDs in the assistant message's opaque `ProviderData`. Later
requests use the most recent ID as `previous_response_id` and send only the
messages after that response. The package no longer sends Chat Completions API
requests.

Constructor options:

- `WithAPIKey(key string)`
- `WithBaseURL(url string)`
- `WithDefaultModel(model string)`
- `WithRequestOption(opt)`

`ChatParams.Options` mappings:

- `previous_response_id` -> explicit stateful continuation override
- `prompt_cache_key` -> stable prompt-cache routing key
- `reasoning_effort` -> compatibility alias for generic reasoning effort
- `reasoning_mode` -> compatibility alias for generic reasoning mode
- `max_output_tokens` (and the `max_tokens` compatibility alias) -> output limit
- `temperature` and `top_p` -> sampling controls
- `parallel_tool_calls` -> enable or disable parallel function calls
- `response_format` -> `text` or `json_object` output formatting
- `strict_tools` -> opt all function tools into OpenAI strict mode

Unknown option keys are ignored safely without failing the run. GPT-5.6 pro
mode is configured provider-neutrally with
`WithReasoning(ReasoningOptions{Mode: "pro", Effort: "..."})`; mode and effort
are sent together and remain independent.

The generic `ChatParams.PreviousResponseID` takes precedence over the
`previous_response_id` compatibility key. Direct adapter callers may use either;
loop callers should prefer `WithPreviousResponseID` so the external ID is sent
only once.

Function tool schemas are always passed through unchanged. Strict mode is
disabled by default and only enabled when `strict_tools` is `true`; callers are
responsible for supplying schemas that meet OpenAI's strict-mode requirements.
The adapter does not make optional fields required or otherwise rewrite schema
semantics.

Moving the adapter to the SDK's `/v3` module leaves the harness provider and
constructor APIs unchanged, except that `WithRequestOption` now accepts
`option.RequestOption` from `github.com/openai/openai-go/v3/option`. Callers
using that SDK escape hatch must update their import path.

Both streaming and non-streaming calls return the terminal response ID, the
same assembled assistant text and function calls, normalized finish reasons,
and cache-aware token usage. OpenAI's inclusive input total is split into
ordinary input, `cached_tokens` reads, and `cache_write_tokens` creation so each
category can be priced directly. Function calls use the API's `call_id` as the
harness `ToolCall.ID`, so generic `RoleTool` results map directly to
`function_call_output` continuation items. Stateful continuations retain the
same `prompt_cache_key` while adding `previous_response_id`.

## Anthropic Adapter

Target SDK: `github.com/anthropics/anthropic-sdk-go`

Status: implemented with equivalent accumulated and delta-streaming caller modes. The adapter uses the SDK's streaming transport in both cases so large output limits are not rejected by the SDK's long-running request guard.

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
- `reasoning_effort` -> compatibility alias for generic reasoning effort
- `output_effort` -> Anthropic-specific output-effort override
- `prompt_cache` -> enable or disable ephemeral prompt caching (enabled by default)
- `cache_ttl` -> `5m` (the default) or `1h`

Adapter-specific mapping considerations:

- System prompt is a top-level request field, not a normal message.
- Tool result responses are represented as user-role tool-result blocks by the SDK.
- Thinking content is mapped to `Message.Thinking` when present.
- Fable is the default model; current Fable, Opus, and Sonnet model IDs can be selected per run.
- Models that require an explicit adaptive-thinking request receive one automatically.
- Signed thinking and continuation blocks are retained in `Message.ProviderData` and replayed unchanged on later tool-loop calls.
- Ephemeral prompt caching remains enabled on every request by default, and
  `cache_ttl` selects its 5-minute or 1-hour TTL without collapsing response
  usage categories.
- Response IDs, stop reasons, refusal details, uncached input, aggregate cache
  creation, 5-minute creation, 1-hour creation, cache reads, and output usage
  are mapped to the shared result contract.

## Testing Approach

- Use `httptest.NewServer` to return canned provider responses
- Verify message and tool-call mapping in both directions
- Verify streaming delta conversion and final accumulation
- Verify usage mapping and error handling semantics

## Design Guidance For Callers

- Prefer adapter constructor options for static provider config and reserve `ChatParams.Options` for per-request tuning.
- Keep one harness call path for all providers; provider differences should stay inside adapters.
- Add integration tests per adapter before enabling by default in examples or docs.
