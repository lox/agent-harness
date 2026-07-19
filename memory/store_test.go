package memory

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	harness "github.com/lox/agent-harness"
)

func TestPromptSectionLoadsRootTodayAndYesterday(t *testing.T) {
	store := newTestStore(t)
	writeFile(t, store, RootFile, "# Durable\n\n- Likes terse summaries.\n")
	writeFile(t, store, "memory/2026-05-22.md", "yesterday note\n")
	writeFile(t, store, "memory/2026-05-23.md", "today note\n")
	writeFile(t, store, "memory/2026-05-21.md", "older note\n")

	section, err := store.PromptSection(context.Background())
	if err != nil {
		t.Fatalf("PromptSection() error = %v", err)
	}

	for _, want := range []string{"MEMORY.md", "Likes terse summaries", "today note", "yesterday note"} {
		if !strings.Contains(section, want) {
			t.Fatalf("PromptSection() missing %q:\n%s", want, section)
		}
	}
	if strings.Contains(section, "older note") {
		t.Fatalf("PromptSection() loaded older dated note:\n%s", section)
	}
}

func TestPromptAndSearchTruncationPreserveUTF8(t *testing.T) {
	store, err := New(t.TempDir(), WithBootstrapByteLimit(3), WithClock(func() time.Time {
		return time.Date(2026, 5, 23, 12, 0, 0, 0, time.UTC)
	}))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	writeFile(t, store, RootFile, strings.Repeat("é", 1000)+" memory")

	section, err := store.PromptSection(context.Background())
	if err != nil {
		t.Fatalf("PromptSection() error = %v", err)
	}
	if !utf8.ValidString(section) {
		t.Fatalf("PromptSection() returned invalid UTF-8: %q", section)
	}
	results, err := store.Search(context.Background(), "memory", SearchOptions{MaxResults: 1})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 || !utf8.ValidString(results[0].Excerpt) {
		t.Fatalf("Search() results = %+v, want valid UTF-8 excerpt", results)
	}
}

func TestSearchFindsMemoryChunks(t *testing.T) {
	store := newTestStore(t)
	writeFile(t, store, RootFile, "# Preferences\n\nThe user prefers conventional commit messages.\n")
	writeFile(t, store, "memory/2026-05-23.md", "# Work\n\nWe discussed vector memory later.\n")

	results, err := store.Search(context.Background(), "conventional commit", SearchOptions{MaxResults: 3})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("Search() returned no results")
	}
	if results[0].Path != RootFile {
		t.Fatalf("first result path = %q, want %q", results[0].Path, RootFile)
	}
	if !strings.Contains(results[0].Excerpt, "conventional commit") {
		t.Fatalf("first result excerpt = %q", results[0].Excerpt)
	}
}

func TestGetRejectsTraversalAndReadsLineRange(t *testing.T) {
	store := newTestStore(t)
	writeFile(t, store, "memory/2026-05-23.md", "one\ntwo\nthree\n")
	writeFile(t, store, "README.md", "not memory\n")

	got, err := store.Get(context.Background(), "memory/2026-05-23.md", 2, 3)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got.Content != "two\nthree" {
		t.Fatalf("Get() content = %q", got.Content)
	}

	if _, err := store.Get(context.Background(), "../secret.md", 0, 0); err == nil {
		t.Fatalf("Get() traversal error = nil, want non-nil")
	}
	if _, err := store.Get(context.Background(), "README.md", 0, 0); err == nil {
		t.Fatalf("Get() non-memory path error = nil, want non-nil")
	}
}

func TestMemoryReadsAndWritesCannotFollowSymlinksOutsideWorkspace(t *testing.T) {
	store := newTestStore(t)
	externalDir := t.TempDir()
	secretPath := filepath.Join(externalDir, "secret.md")
	if err := os.WriteFile(secretPath, []byte("secret\n"), 0o644); err != nil {
		t.Fatalf("write external secret: %v", err)
	}
	readLink := filepath.Join(store.WorkspaceDir(), "memory", "linked.md")
	if err := os.Symlink(secretPath, readLink); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := store.Get(context.Background(), "memory/linked.md", 0, 0); err == nil {
		t.Fatal("Get() followed a symlink outside the workspace")
	}

	dailyPath := filepath.Join(store.WorkspaceDir(), "memory", "2026-05-23.md")
	if err := os.Symlink(secretPath, dailyPath); err != nil {
		t.Fatalf("create daily memory symlink: %v", err)
	}
	if _, err := store.AppendDaily(context.Background(), "Manual", "overwrite"); err == nil {
		t.Fatal("AppendDaily() followed a symlink outside the workspace")
	}
	content, err := os.ReadFile(secretPath)
	if err != nil {
		t.Fatalf("read external secret: %v", err)
	}
	if string(content) != "secret\n" {
		t.Fatalf("external file changed: %q", content)
	}
}

func TestToolsReturnJSONPayloads(t *testing.T) {
	store := newTestStore(t)
	writeFile(t, store, RootFile, "Durable preference: terse summaries.\n")

	tool := store.SearchTool()
	res, err := tool.Execute(context.Background(), harness.ToolCall{
		ID:        "call-1",
		Name:      "memory_search",
		Arguments: json.RawMessage(`{"query":"terse summaries","max_results":1}`),
	})
	if err != nil {
		t.Fatalf("memory_search Execute() error = %v", err)
	}
	if res.ToolCallID != "call-1" {
		t.Fatalf("ToolCallID = %q", res.ToolCallID)
	}
	if !strings.Contains(res.Content, `"provider": "builtin-lexical"`) {
		t.Fatalf("memory_search content = %s", res.Content)
	}
}

func TestSearchToolRecordsRecallSignalsForWorkingMemory(t *testing.T) {
	store := newTestStore(t)
	writeFile(t, store, "memory/2026-05-23.md", "Decision: build pluggable promotion primitives.\n")

	tool := store.SearchTool()
	for _, query := range []string{"pluggable promotion", "promotion primitives"} {
		if _, err := tool.Execute(context.Background(), harness.ToolCall{
			ID:        "call-" + query,
			Name:      "memory_search",
			Arguments: json.RawMessage(`{"query":` + strconvQuote(query) + `}`),
		}); err != nil {
			t.Fatalf("memory_search Execute(%q) error = %v", query, err)
		}
	}

	candidates, err := store.PromotionCandidates(context.Background(), PromotionOptions{
		MinRecallCount:   2,
		MinUniqueQueries: 2,
	})
	if err != nil {
		t.Fatalf("PromotionCandidates() error = %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("PromotionCandidates() len = %d, want 1: %#v", len(candidates), candidates)
	}
	if candidates[0].Path != "memory/2026-05-23.md" {
		t.Fatalf("candidate path = %q", candidates[0].Path)
	}
	if candidates[0].RecallCount != 2 || candidates[0].UniqueQueries != 2 {
		t.Fatalf("candidate signals = recalls %d queries %d", candidates[0].RecallCount, candidates[0].UniqueQueries)
	}
}

func TestApplyPromotionsWritesRootMemoryAndMarksSignals(t *testing.T) {
	store := newTestStore(t)
	writeFile(t, store, "memory/2026-05-23.md", "Decision: keep durable memory source-grounded and reviewable.\n")

	results, err := store.Search(context.Background(), "durable source grounded", SearchOptions{MaxResults: 1})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	for _, query := range []string{"durable source grounded", "reviewable memory"} {
		if err := store.RecordSearchResults(context.Background(), query, results); err != nil {
			t.Fatalf("RecordSearchResults(%q) error = %v", query, err)
		}
	}

	consolidator := NewConsolidator(store, store)
	candidates, err := consolidator.Preview(context.Background(), PromotionOptions{
		MinRecallCount:   2,
		MinUniqueQueries: 2,
	})
	if err != nil {
		t.Fatalf("Preview() error = %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("Preview() len = %d, want 1", len(candidates))
	}

	result, err := consolidator.Apply(context.Background(), PromotionOptions{
		MinRecallCount:   2,
		MinUniqueQueries: 2,
	}, ApplyOptions{Heading: "Promoted Test Memory"})
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	if len(result.Applied) != 1 {
		t.Fatalf("Apply() applied = %d, want 1: %#v", len(result.Applied), result)
	}

	root := readFile(t, store, RootFile)
	for _, want := range []string{
		"## Promoted Test Memory (2026-05-23)",
		"agent-harness-memory-promotion:",
		"Source: `memory/2026-05-23.md:",
		"source-grounded and reviewable",
	} {
		if !strings.Contains(root, want) {
			t.Fatalf("MEMORY.md missing %q:\n%s", want, root)
		}
	}

	after, err := consolidator.Apply(context.Background(), PromotionOptions{
		MinRecallCount:   2,
		MinUniqueQueries: 2,
	}, ApplyOptions{Heading: "Promoted Test Memory"})
	if err != nil {
		t.Fatalf("second Apply() error = %v", err)
	}
	if len(after.Applied) != 0 {
		t.Fatalf("second Apply() applied = %d, want 0", len(after.Applied))
	}
}

func TestApplyPromotionsSkipsChangedSourceExcerpt(t *testing.T) {
	store := newTestStore(t)
	const source = "memory/2026-05-23.md"
	writeFile(t, store, source, "Decision: keep the reviewed text.\n")
	results, err := store.Search(context.Background(), "reviewed text", SearchOptions{MaxResults: 1})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	for _, query := range []string{"reviewed text", "keep reviewed"} {
		if err := store.RecordSearchResults(context.Background(), query, results); err != nil {
			t.Fatalf("RecordSearchResults(%q) error = %v", query, err)
		}
	}
	candidates, err := store.PromotionCandidates(context.Background(), PromotionOptions{MinRecallCount: 2, MinUniqueQueries: 2})
	if err != nil || len(candidates) != 1 {
		t.Fatalf("PromotionCandidates() = %+v, %v", candidates, err)
	}

	writeFile(t, store, source, "Decision: this text was not reviewed.\n")
	result, err := store.ApplyPromotions(context.Background(), candidates, ApplyOptions{})
	if err != nil {
		t.Fatalf("ApplyPromotions() error = %v", err)
	}
	if len(result.Applied) != 0 || len(result.Skipped) != 1 || result.Skipped[0].Reason != "source unavailable" {
		t.Fatalf("ApplyPromotions() result = %+v, want changed source skipped", result)
	}
}

func TestAppendDailyAndCaptureThread(t *testing.T) {
	store := newTestStore(t)

	relPath, err := store.AppendDaily(context.Background(), "Preference", "Use file-backed memory.")
	if err != nil {
		t.Fatalf("AppendDaily() error = %v", err)
	}
	if relPath != "memory/2026-05-23.md" {
		t.Fatalf("AppendDaily() path = %q", relPath)
	}

	thread := harness.NewThread()
	thread.AddUser("remember this")
	thread.Append(&harness.Result{Messages: []harness.Message{{Role: harness.RoleAssistant, Content: "stored"}}})

	captured, err := store.CaptureThread(context.Background(), thread, CaptureOptions{
		Title:              "Reset",
		Slug:               "Before Reset",
		IncludeToolResults: true,
	})
	if err != nil {
		t.Fatalf("CaptureThread() error = %v", err)
	}
	if captured != "memory/2026-05-23-120000-before-reset.md" {
		t.Fatalf("CaptureThread() path = %q", captured)
	}
	content := readFile(t, store, captured)
	if !strings.Contains(content, "remember this") || !strings.Contains(content, "stored") {
		t.Fatalf("captured content = %s", content)
	}
}

func TestConcurrentCapturesClaimUniqueFiles(t *testing.T) {
	store := newTestStore(t)
	messages := []harness.Message{{Role: harness.RoleUser, Content: "remember each capture"}}

	const captures = 16
	var wg sync.WaitGroup
	paths := make(chan string, captures)
	errs := make(chan error, captures)
	start := make(chan struct{})
	for range captures {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			captured, err := store.CaptureMessages(context.Background(), messages, CaptureOptions{Slug: "parallel"})
			if err != nil {
				errs <- err
				return
			}
			paths <- captured
		}()
	}
	close(start)
	wg.Wait()
	close(paths)
	close(errs)

	for err := range errs {
		t.Fatalf("CaptureMessages() error = %v", err)
	}
	unique := make(map[string]struct{}, captures)
	for captured := range paths {
		unique[captured] = struct{}{}
	}
	if len(unique) != captures {
		t.Fatalf("unique capture paths = %d, want %d: %v", len(unique), captures, unique)
	}
}

func newTestStore(t *testing.T) *Store {
	t.Helper()

	store, err := New(t.TempDir(), WithClock(func() time.Time {
		return time.Date(2026, 5, 23, 12, 0, 0, 0, time.UTC)
	}))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := store.Ensure(context.Background()); err != nil {
		t.Fatalf("Ensure() error = %v", err)
	}
	return store
}

func writeFile(t *testing.T, store *Store, relPath, content string) {
	t.Helper()

	absPath := filepath.Join(store.WorkspaceDir(), filepath.FromSlash(relPath))
	if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", relPath, err)
	}
	if err := os.WriteFile(absPath, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", relPath, err)
	}
}

func readFile(t *testing.T, store *Store, relPath string) string {
	t.Helper()

	absPath := filepath.Join(store.WorkspaceDir(), filepath.FromSlash(relPath))
	content, err := os.ReadFile(absPath)
	if err != nil {
		t.Fatalf("read %s: %v", relPath, err)
	}
	return string(content)
}

func strconvQuote(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(b)
}
