# Provider Adapters

This document tracks the provider adapter strategy for `agent-harness`.

## Adapter Principles

- Adapters implement `harness.Provider`
- Keep adapter code thin: map harness types to SDK types and back
- Keep provider-specific settings in adapter options
- Support both non-streaming and streaming paths
- Keep adapter tests network-free via local HTTP servers

## Planned Packages

```
provider/
├── openai/
└── anthropic/
```

## OpenAI Adapter

Target SDK: `github.com/openai/openai-go`

Planned constructor options:

- `WithAPIKey(key string)`
- `WithBaseURL(url string)`
- `WithDefaultModel(model string)`

Planned `ChatParams.Options` mappings:

- `temperature`
- `max_tokens`
- `top_p`
- `reasoning_effort`
- `response_format`

## Anthropic Adapter

Target SDK: `github.com/anthropics/anthropic-sdk-go`

Planned constructor options:

- `WithAPIKey(key string)`
- `WithDefaultModel(model string)`
- `WithPromptCaching(enabled bool)`

Planned `ChatParams.Options` mappings:

- `temperature`
- `max_tokens`
- `top_p`
- `top_k`
- `thinking_budget`

## Testing Approach

- Use `httptest.NewServer` to return canned provider responses
- Verify message and tool-call mapping in both directions
- Verify streaming delta conversion and final accumulation
- Verify usage mapping and error handling semantics
