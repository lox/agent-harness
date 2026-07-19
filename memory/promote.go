package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	// RecallSignalPath stores auxiliary recall signals used by the default
	// promotion source.
	RecallSignalPath = DailyDir + "/.signals/recalls.json"

	defaultPromotionMaxCandidates    = 8
	defaultPromotionMinRecallCount   = 3
	defaultPromotionMinUniqueQueries = 2
	defaultPromotionMinScore         = 0.65
	promotionMarkerPrefix            = "agent-harness-memory-promotion:"
)

// RecallSignal records that a working-memory excerpt was useful during recall.
type RecallSignal struct {
	Key             string     `json:"key"`
	Path            string     `json:"path"`
	StartLine       int        `json:"start_line"`
	EndLine         int        `json:"end_line"`
	Excerpt         string     `json:"excerpt"`
	RecallCount     int        `json:"recall_count"`
	TotalScore      float64    `json:"total_score"`
	MaxScore        float64    `json:"max_score"`
	FirstRecalledAt time.Time  `json:"first_recalled_at"`
	LastRecalledAt  time.Time  `json:"last_recalled_at"`
	QueryHashes     []string   `json:"query_hashes,omitempty"`
	PromotedAt      *time.Time `json:"promoted_at,omitempty"`
}

// PromotionOptions controls how recall signals become promotion candidates.
type PromotionOptions struct {
	// MinScore is the minimum normalized promotion score. Values <= 0 use a
	// conservative default.
	MinScore float64

	// MinRecallCount is the minimum number of recorded recalls required.
	// Values <= 0 use a conservative default.
	MinRecallCount int

	// MinUniqueQueries is the minimum number of distinct search queries
	// required. Values <= 0 use a conservative default.
	MinUniqueQueries int

	// MaxCandidates limits returned candidates. Values <= 0 use a conservative
	// default.
	MaxCandidates int

	// IncludePromoted includes candidates already marked as promoted in the
	// recall signal store.
	IncludePromoted bool
}

// PromotionCandidate is a source-grounded memory excerpt eligible for durable
// promotion.
type PromotionCandidate struct {
	Key            string    `json:"key"`
	Path           string    `json:"path"`
	StartLine      int       `json:"start_line"`
	EndLine        int       `json:"end_line"`
	Excerpt        string    `json:"excerpt"`
	Score          float64   `json:"score"`
	RecallCount    int       `json:"recall_count"`
	UniqueQueries  int       `json:"unique_queries"`
	LastRecalledAt time.Time `json:"last_recalled_at"`
	Reasons        []string  `json:"reasons,omitempty"`
}

// ApplyOptions controls how promotions are written to durable memory.
type ApplyOptions struct {
	// Heading is the section heading appended to MEMORY.md.
	Heading string
}

// PromotionSkip describes a candidate that was not applied.
type PromotionSkip struct {
	Candidate PromotionCandidate `json:"candidate"`
	Reason    string             `json:"reason"`
}

// PromotionResult describes an apply operation.
type PromotionResult struct {
	Path    string               `json:"path"`
	Applied []PromotionCandidate `json:"applied"`
	Skipped []PromotionSkip      `json:"skipped,omitempty"`
}

// CandidateSource provides promotion candidates. LLM, vector, or external
// memory systems can implement this interface without changing the harness.
type CandidateSource interface {
	PromotionCandidates(ctx context.Context, opts PromotionOptions) ([]PromotionCandidate, error)
}

// PromotionApplier writes selected candidates to durable memory.
type PromotionApplier interface {
	ApplyPromotions(ctx context.Context, candidates []PromotionCandidate, opts ApplyOptions) (*PromotionResult, error)
}

// Consolidator combines a candidate source with a promotion applier.
type Consolidator struct {
	Source  CandidateSource
	Applier PromotionApplier
}

// NewConsolidator returns a pluggable promotion pipeline. A Store can be used
// as both source and applier for the built-in file-backed behavior.
func NewConsolidator(source CandidateSource, applier PromotionApplier) *Consolidator {
	return &Consolidator{Source: source, Applier: applier}
}

// Preview returns candidates selected by the configured source.
func (c *Consolidator) Preview(ctx context.Context, opts PromotionOptions) ([]PromotionCandidate, error) {
	if c == nil || c.Source == nil {
		return nil, errors.New("memory consolidator source is nil")
	}
	return c.Source.PromotionCandidates(ctx, opts)
}

// Apply previews candidates and writes the selected set with the configured
// applier.
func (c *Consolidator) Apply(ctx context.Context, opts PromotionOptions, applyOpts ApplyOptions) (*PromotionResult, error) {
	if c == nil || c.Applier == nil {
		return nil, errors.New("memory consolidator applier is nil")
	}
	candidates, err := c.Preview(ctx, opts)
	if err != nil {
		return nil, err
	}
	return c.Applier.ApplyPromotions(ctx, candidates, applyOpts)
}

// RecordSearchResults records recall signals for working-memory search hits.
// Durable MEMORY.md hits are intentionally ignored because they are already
// promoted.
func (s *Store) RecordSearchResults(ctx context.Context, query string, results []SearchResult) error {
	if err := s.Ensure(ctx); err != nil {
		return err
	}
	query = strings.TrimSpace(query)
	if query == "" || len(results) == 0 {
		return nil
	}

	now := s.now().UTC()
	queryHash := hashText(strings.ToLower(query))

	s.mu.Lock()
	defer s.mu.Unlock()

	signals, err := s.readRecallSignalsLocked()
	if err != nil {
		return err
	}

	changed := false
	for _, result := range results {
		if !isPromotableMemoryPath(result.Path) {
			continue
		}
		excerpt := strings.TrimSpace(result.Excerpt)
		if excerpt == "" {
			continue
		}

		key := signalKey(result.Path, result.StartLine, result.EndLine, excerpt)
		signal := signals[key]
		if signal.Key == "" {
			signal = RecallSignal{
				Key:             key,
				Path:            filepathSlash(result.Path),
				StartLine:       result.StartLine,
				EndLine:         result.EndLine,
				Excerpt:         excerpt,
				FirstRecalledAt: now,
			}
		}
		signal.RecallCount++
		signal.TotalScore += result.Score
		if result.Score > signal.MaxScore {
			signal.MaxScore = result.Score
		}
		signal.LastRecalledAt = now
		signal.QueryHashes = appendUnique(signal.QueryHashes, queryHash)
		sort.Strings(signal.QueryHashes)
		signals[key] = signal
		changed = true
	}

	if !changed {
		return nil
	}
	return s.writeRecallSignalsLocked(signals)
}

// PromotionCandidates ranks recall signals into source-grounded promotion
// candidates.
func (s *Store) PromotionCandidates(ctx context.Context, opts PromotionOptions) ([]PromotionCandidate, error) {
	if err := s.Ensure(ctx); err != nil {
		return nil, err
	}
	opts = normalizePromotionOptions(opts)

	s.mu.Lock()
	signals, err := s.readRecallSignalsLocked()
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}

	now := s.now().UTC()
	candidates := make([]PromotionCandidate, 0, len(signals))
	for _, signal := range signals {
		if !opts.IncludePromoted && signal.PromotedAt != nil {
			continue
		}
		if !isPromotableMemoryPath(signal.Path) {
			continue
		}
		if signal.RecallCount < opts.MinRecallCount {
			continue
		}
		uniqueQueries := len(signal.QueryHashes)
		if uniqueQueries < opts.MinUniqueQueries {
			continue
		}

		excerpt, err := s.rehydratePromotionExcerpt(ctx, signal)
		if err != nil {
			continue
		}

		score := scorePromotionSignal(signal, opts, now)
		if score < opts.MinScore {
			continue
		}

		candidates = append(candidates, PromotionCandidate{
			Key:            signal.Key,
			Path:           signal.Path,
			StartLine:      signal.StartLine,
			EndLine:        signal.EndLine,
			Excerpt:        excerpt,
			Score:          score,
			RecallCount:    signal.RecallCount,
			UniqueQueries:  uniqueQueries,
			LastRecalledAt: signal.LastRecalledAt,
			Reasons: []string{
				fmt.Sprintf("recalls=%d", signal.RecallCount),
				fmt.Sprintf("unique_queries=%d", uniqueQueries),
				fmt.Sprintf("source=%s:%d-%d", signal.Path, signal.StartLine, signal.EndLine),
			},
		})
	}

	sortPromotionCandidates(candidates)
	if len(candidates) > opts.MaxCandidates {
		candidates = candidates[:opts.MaxCandidates]
	}
	return candidates, nil
}

// ApplyPromotions appends selected candidates to MEMORY.md and marks their
// recall signals as promoted.
func (s *Store) ApplyPromotions(ctx context.Context, candidates []PromotionCandidate, opts ApplyOptions) (*PromotionResult, error) {
	if err := s.Ensure(ctx); err != nil {
		return nil, err
	}
	result := &PromotionResult{Path: RootFile}
	if len(candidates) == 0 {
		return result, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	signals, err := s.readRecallSignalsLocked()
	if err != nil {
		return nil, err
	}

	root, err := s.openRoot()
	if err != nil {
		return nil, err
	}
	defer root.Close()
	rootContent, err := root.ReadFile(RootFile)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("read %s: %w", RootFile, err)
	}

	sortPromotionCandidates(candidates)
	now := s.now().UTC()
	applied := make([]PromotionCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Key == "" {
			result.Skipped = append(result.Skipped, PromotionSkip{Candidate: candidate, Reason: "missing key"})
			continue
		}
		if strings.Contains(string(rootContent), promotionMarker(candidate.Key)) {
			result.Skipped = append(result.Skipped, PromotionSkip{Candidate: candidate, Reason: "already promoted"})
			s.markPromotedSignal(signals, candidate.Key, now)
			continue
		}
		signal, ok := signals[candidate.Key]
		if !ok {
			result.Skipped = append(result.Skipped, PromotionSkip{Candidate: candidate, Reason: "missing recall signal"})
			continue
		}
		if signal.PromotedAt != nil {
			result.Skipped = append(result.Skipped, PromotionSkip{Candidate: candidate, Reason: "already marked promoted"})
			continue
		}
		excerpt, err := s.rehydratePromotionExcerpt(ctx, signal)
		if err != nil {
			result.Skipped = append(result.Skipped, PromotionSkip{Candidate: candidate, Reason: "source unavailable"})
			continue
		}
		candidate.Excerpt = excerpt
		applied = append(applied, candidate)
	}

	if len(applied) == 0 {
		if err := s.writeRecallSignalsLocked(signals); err != nil {
			return nil, err
		}
		result.Applied = nil
		return result, nil
	}

	section := buildPromotionSection(now, applied, opts)
	f, err := root.OpenFile(RootFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", RootFile, err)
	}
	if _, err := f.WriteString(section); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("write %s: %w", RootFile, err)
	}
	if err := f.Close(); err != nil {
		return nil, fmt.Errorf("close %s: %w", RootFile, err)
	}

	for _, candidate := range applied {
		s.markPromotedSignal(signals, candidate.Key, now)
	}
	if err := s.writeRecallSignalsLocked(signals); err != nil {
		return nil, err
	}

	result.Applied = applied
	return result, nil
}

type recallSignalFile struct {
	Version int            `json:"version"`
	Items   []RecallSignal `json:"items"`
}

func (s *Store) readRecallSignalsLocked() (map[string]RecallSignal, error) {
	root, err := s.openRoot()
	if err != nil {
		return nil, err
	}
	defer root.Close()
	content, err := root.ReadFile(RecallSignalPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return map[string]RecallSignal{}, nil
		}
		return nil, fmt.Errorf("read recall signals: %w", err)
	}
	var file recallSignalFile
	if err := json.Unmarshal(content, &file); err != nil {
		return nil, fmt.Errorf("parse recall signals: %w", err)
	}
	signals := make(map[string]RecallSignal, len(file.Items))
	for _, signal := range file.Items {
		if signal.Key == "" {
			signal.Key = signalKey(signal.Path, signal.StartLine, signal.EndLine, signal.Excerpt)
		}
		signals[signal.Key] = signal
	}
	return signals, nil
}

func (s *Store) writeRecallSignalsLocked(signals map[string]RecallSignal) error {
	items := make([]RecallSignal, 0, len(signals))
	for _, signal := range signals {
		items = append(items, signal)
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].Path == items[j].Path {
			if items[i].StartLine == items[j].StartLine {
				return items[i].Key < items[j].Key
			}
			return items[i].StartLine < items[j].StartLine
		}
		return items[i].Path < items[j].Path
	})

	content, err := json.MarshalIndent(recallSignalFile{Version: 1, Items: items}, "", "  ")
	if err != nil {
		return fmt.Errorf("encode recall signals: %w", err)
	}
	root, err := s.openRoot()
	if err != nil {
		return err
	}
	defer root.Close()
	if err := root.MkdirAll(filepathSlash(filepath.Dir(RecallSignalPath)), 0o755); err != nil {
		return fmt.Errorf("create recall signal directory: %w", err)
	}
	if err := root.WriteFile(RecallSignalPath, append(content, '\n'), 0o644); err != nil {
		return fmt.Errorf("write recall signals: %w", err)
	}
	return nil
}

func (s *Store) rehydratePromotionExcerpt(ctx context.Context, signal RecallSignal) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	got, err := s.Get(ctx, signal.Path, signal.StartLine, signal.EndLine)
	if err != nil {
		return "", err
	}
	text := strings.TrimSpace(got.Content)
	if text == "" {
		return "", errors.New("empty promotion excerpt")
	}
	current := excerpt(text)
	if current != strings.TrimSpace(signal.Excerpt) {
		return "", errors.New("promotion source excerpt changed")
	}
	return current, nil
}

func (s *Store) markPromotedSignal(signals map[string]RecallSignal, key string, at time.Time) {
	signal, ok := signals[key]
	if !ok {
		return
	}
	promotedAt := at
	signal.PromotedAt = &promotedAt
	signals[key] = signal
}

func normalizePromotionOptions(opts PromotionOptions) PromotionOptions {
	if opts.MinScore <= 0 {
		opts.MinScore = defaultPromotionMinScore
	}
	if opts.MinRecallCount <= 0 {
		opts.MinRecallCount = defaultPromotionMinRecallCount
	}
	if opts.MinUniqueQueries <= 0 {
		opts.MinUniqueQueries = defaultPromotionMinUniqueQueries
	}
	if opts.MaxCandidates <= 0 {
		opts.MaxCandidates = defaultPromotionMaxCandidates
	}
	return opts
}

func scorePromotionSignal(signal RecallSignal, opts PromotionOptions, now time.Time) float64 {
	recallComponent := clamp01(float64(signal.RecallCount) / float64(opts.MinRecallCount))
	diversityComponent := clamp01(float64(len(signal.QueryHashes)) / float64(opts.MinUniqueQueries))
	averageScore := 0.0
	if signal.RecallCount > 0 {
		averageScore = signal.TotalScore / float64(signal.RecallCount)
	}
	qualityComponent := averageScore / (averageScore + 4)
	if math.IsNaN(qualityComponent) || math.IsInf(qualityComponent, 0) {
		qualityComponent = 0
	}
	recencyComponent := 1.0
	if !signal.LastRecalledAt.IsZero() {
		age := now.Sub(signal.LastRecalledAt)
		if age < 0 {
			age = 0
		}
		recencyComponent = math.Pow(0.5, age.Hours()/24/14)
	}
	score := 0.45*recallComponent + 0.25*diversityComponent + 0.20*clamp01(qualityComponent) + 0.10*clamp01(recencyComponent)
	return math.Round(score*1000) / 1000
}

func sortPromotionCandidates(candidates []PromotionCandidate) {
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Score == candidates[j].Score {
			if candidates[i].RecallCount == candidates[j].RecallCount {
				if candidates[i].Path == candidates[j].Path {
					return candidates[i].StartLine < candidates[j].StartLine
				}
				return candidates[i].Path < candidates[j].Path
			}
			return candidates[i].RecallCount > candidates[j].RecallCount
		}
		return candidates[i].Score > candidates[j].Score
	})
}

func buildPromotionSection(now time.Time, candidates []PromotionCandidate, opts ApplyOptions) string {
	heading := strings.TrimSpace(opts.Heading)
	if heading == "" {
		heading = "Promoted From Working Memory"
	}

	var b strings.Builder
	b.WriteString("\n\n## ")
	b.WriteString(heading)
	b.WriteString(" (")
	b.WriteString(now.Format("2006-01-02"))
	b.WriteString(")\n")

	for _, candidate := range candidates {
		b.WriteString("\n<!-- ")
		b.WriteString(promotionMarkerPrefix)
		b.WriteString(candidate.Key)
		b.WriteString(" -->\n")
		b.WriteString("Source: `")
		b.WriteString(candidate.Path)
		b.WriteString(":")
		b.WriteString(fmt.Sprintf("%d-%d", candidate.StartLine, candidate.EndLine))
		b.WriteString("`; score=")
		b.WriteString(fmt.Sprintf("%.3f", candidate.Score))
		b.WriteString("; recalls=")
		b.WriteString(fmt.Sprintf("%d", candidate.RecallCount))
		b.WriteString("; unique_queries=")
		b.WriteString(fmt.Sprintf("%d", candidate.UniqueQueries))
		b.WriteString("\n\n")
		for _, line := range splitLines(candidate.Excerpt) {
			b.WriteString("> ")
			b.WriteString(line)
			b.WriteString("\n")
		}
	}
	return b.String()
}

func isPromotableMemoryPath(path string) bool {
	clean, err := cleanMemoryFilePath(path)
	if err != nil {
		return false
	}
	if clean == RootFile || !strings.HasPrefix(clean, DailyDir+"/") {
		return false
	}
	for _, part := range strings.Split(strings.TrimPrefix(clean, DailyDir+"/"), "/") {
		if strings.HasPrefix(part, ".") {
			return false
		}
	}
	return true
}

func signalKey(path string, startLine, endLine int, excerpt string) string {
	return hashText(fmt.Sprintf("%s:%d:%d:%s", filepathSlash(path), startLine, endLine, strings.TrimSpace(excerpt)))
}

func hashText(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])[:16]
}

func appendUnique(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func promotionMarker(key string) string {
	return promotionMarkerPrefix + key
}

func clamp01(v float64) float64 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}
