# Memory

The `memory` package provides an optional OpenClaw-style memory layer for
applications built on agent-harness.

The core harness stays stateless. Memory is a composable package that an
application wires into prompts, tools, and lifecycle commands.

## File Contract

Memory is stored as Markdown under a workspace directory.

```text
<workspace>/
+-- MEMORY.md
`-- memory/
    +-- 2026-05-23.md
    `-- 2026-05-23-120000-reset.md
```

- `MEMORY.md` is curated long-term memory.
- `memory/YYYY-MM-DD.md` is the daily working memory file.
- `memory/YYYY-MM-DD-HHMMSS-<slug>.md` captures session or pre-compaction
  context.

Markdown files are the source of truth. Search results are derived from those
files and can be rebuilt by reading the workspace again.

All package I/O is rooted at the configured workspace. Symlinks may point
within that workspace, but reads and writes reject links that escape it.

## Prompt Bootstrap

Use `Store.Bootstrap` to append a memory prompt section to your system prompt.
The section loads:

- `MEMORY.md`
- today's dated memory files
- yesterday's dated memory files

Bootstrap content is limited to 64 KiB by default. When the limit is reached,
the prompt tells the model to use `memory_search` and `memory_get` for the
remaining context. Pass `memory.WithBootstrapByteLimit(limit)` to change the
limit, or use a value of `0` or less to disable truncation.

The prompt also tells the model to use `memory_search` before answering from
prior work, preferences, decisions, plans, people, dates, or stored context.

```go
store, err := memory.New("/path/to/workspace")
if err != nil {
    return err
}

system, err := store.Bootstrap(ctx, "You are a concise assistant.")
if err != nil {
    return err
}

tools := append([]harness.Tool(nil), appTools...)
tools = append(tools, store.Tools()...)

result, err := harness.Run(ctx, provider,
    harness.WithSystem(system),
    harness.WithMessages(thread.Messages...),
    harness.WithTools(tools...),
)
```

## Recall Tools

`Store.Tools()` returns:

- `memory_search`: lexical search over `MEMORY.md` and `memory/*.md`
- `memory_get`: exact line-range reads from memory files

The first implementation uses built-in lexical scoring. The package boundary is
intended to support SQLite, vector search, or hybrid indexing later without
changing the Markdown file contract.

`memory_search` also records recall signals for working-memory results under
`memory/.signals/recalls.json`. Hits from `MEMORY.md` are not recorded because
they are already durable memory.

## Write Paths

Use `AppendDaily` for explicit memories:

```go
_, err := store.AppendDaily(ctx, "Preference", "The user prefers conventional commit messages.")
```

Use `CaptureThread` or `CaptureMessages` when starting a new thread, resetting a
session, or flushing context before compaction:

```go
_, err := store.CaptureThread(ctx, thread, memory.CaptureOptions{
    Title:       "Session Memory",
    Slug:        "reset",
    MaxMessages: 15,
})
```

Capture writes recent context to a timestamped file under `memory/`. It does not
promote into `MEMORY.md`; durable promotion should be a separate, reviewable
step. Concurrent captures claim filenames atomically, so equal timestamps do
not overwrite earlier captures. Tool result content is omitted by default; set
`CaptureOptions.IncludeToolResults` when the captured transcript should include
it.

## Consolidation

The package includes a small OpenClaw-style consolidation primitive. It is not a
claim that memory is deterministic end to end. The semantic signal can still come
from model behavior, search usage, or another candidate source. The durable write
path is deterministic, source-grounded, and reviewable.

The built-in flow is:

1. `memory_search` records recall signals for matching `memory/*.md` excerpts.
2. `PromotionCandidates` ranks excerpts by recall count, unique queries, lexical
   quality, and recency.
3. `ApplyPromotions` re-reads the source lines and appends selected excerpts to
   `MEMORY.md` with a stable promotion marker and source provenance.

```go
candidates, err := store.PromotionCandidates(ctx, memory.PromotionOptions{})
if err != nil {
    return err
}

// Show candidates to an operator or apply an application-specific policy here.

result, err := store.ApplyPromotions(ctx, candidates, memory.ApplyOptions{})
if err != nil {
    return err
}
_ = result
```

The default thresholds require repeated recall from distinct queries before a
candidate is selected. Applications can tune `PromotionOptions`, provide their
own `CandidateSource`, or replace the `PromotionApplier` while keeping the file
contract intact. `NewConsolidator(source, applier)` is the convenience wrapper
for that pluggable path.

## Example

`examples/claw` enables memory by default at `~/.agent-harness/claw`.

```bash
OPENAI_API_KEY=... go run ./examples/claw
```

Useful commands:

- `/memory` prints the memory workspace.
- `/remember <text>` appends a manual memory to today's file.
- `/new` captures the current thread to `memory/` and starts a new thread.

Set `--memory-dir ""` to disable memory for the example.
