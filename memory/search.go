package memory

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	defaultMaxResults = 8
	maxSearchResults  = 20
	maxExcerptBytes   = 1600
)

// SearchOptions controls file-backed memory search.
type SearchOptions struct {
	MaxResults int
	MinScore   float64
}

// SearchResult is one matching memory excerpt.
type SearchResult struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float64 `json:"score"`
	Excerpt   string  `json:"excerpt"`
}

// Excerpt is an exact line range from a memory file.
type Excerpt struct {
	Path      string `json:"path"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Content   string `json:"content"`
}

// Search performs lexical recall over MEMORY.md and memory/*.md.
func (s *Store) Search(ctx context.Context, query string, opts SearchOptions) ([]SearchResult, error) {
	if err := s.Ensure(ctx); err != nil {
		return nil, err
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("memory search query cannot be empty")
	}
	if opts.MaxResults <= 0 {
		opts.MaxResults = defaultMaxResults
	}
	if opts.MaxResults > maxSearchResults {
		opts.MaxResults = maxSearchResults
	}

	queryTerms := tokenSet(query)
	if len(queryTerms) == 0 {
		return nil, errors.New("memory search query has no searchable terms")
	}

	root, err := s.openRoot()
	if err != nil {
		return nil, err
	}
	defer root.Close()

	refs, err := s.allMemoryRefs(root)
	if err != nil {
		return nil, err
	}

	var results []SearchResult
	queryLower := strings.ToLower(query)
	for _, ref := range refs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		content, err := root.ReadFile(ref.relPath)
		if err != nil {
			return nil, fmt.Errorf("read memory file %s: %w", ref.relPath, err)
		}
		for _, chunk := range chunkMarkdown(ref.relPath, string(content)) {
			score := scoreChunk(queryTerms, queryLower, chunk.text)
			if score < opts.MinScore || score <= 0 {
				continue
			}
			results = append(results, SearchResult{
				Path:      chunk.path,
				StartLine: chunk.startLine,
				EndLine:   chunk.endLine,
				Score:     score,
				Excerpt:   excerpt(chunk.text),
			})
		}
	}

	sort.SliceStable(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			if results[i].Path == results[j].Path {
				return results[i].StartLine < results[j].StartLine
			}
			return results[i].Path < results[j].Path
		}
		return results[i].Score > results[j].Score
	})
	if len(results) > opts.MaxResults {
		results = results[:opts.MaxResults]
	}
	return results, nil
}

// Get returns an exact 1-indexed line range from a memory file.
func (s *Store) Get(ctx context.Context, relPath string, startLine, endLine int) (*Excerpt, error) {
	if err := s.Ensure(ctx); err != nil {
		return nil, err
	}
	if strings.TrimSpace(relPath) == "" {
		return nil, errors.New("memory path is required")
	}
	if startLine < 0 || endLine < 0 {
		return nil, errors.New("memory line numbers cannot be negative")
	}
	if startLine == 0 {
		startLine = 1
	}

	cleanPath, err := cleanMemoryFilePath(relPath)
	if err != nil {
		return nil, err
	}
	root, err := s.openRoot()
	if err != nil {
		return nil, err
	}
	defer root.Close()
	content, err := root.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("read memory file %s: %w", cleanPath, err)
	}

	lines := splitLines(string(content))
	if len(lines) == 0 {
		return &Excerpt{Path: cleanPath, StartLine: 1, EndLine: 0}, nil
	}
	if endLine == 0 || endLine > len(lines) {
		endLine = len(lines)
	}
	if startLine > len(lines) {
		return nil, fmt.Errorf("start line %d is past end of %s", startLine, cleanPath)
	}
	if startLine > endLine {
		return nil, fmt.Errorf("start line %d is after end line %d", startLine, endLine)
	}

	return &Excerpt{
		Path:      cleanPath,
		StartLine: startLine,
		EndLine:   endLine,
		Content:   strings.Join(lines[startLine-1:endLine], "\n"),
	}, nil
}

type markdownChunk struct {
	path      string
	startLine int
	endLine   int
	text      string
}

func chunkMarkdown(path, content string) []markdownChunk {
	lines := splitLines(content)
	var chunks []markdownChunk
	start := 0
	var current []string

	flush := func(end int) {
		text := strings.TrimSpace(strings.Join(current, "\n"))
		if text != "" {
			chunks = append(chunks, markdownChunk{
				path:      path,
				startLine: start + 1,
				endLine:   end,
				text:      text,
			})
		}
		current = nil
	}

	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			if len(current) > 0 {
				flush(i)
			}
			start = i + 1
			continue
		}
		if len(current) == 0 {
			start = i
		}
		current = append(current, line)
	}
	if len(current) > 0 {
		flush(len(lines))
	}
	return chunks
}

func scoreChunk(queryTerms map[string]int, queryLower, text string) float64 {
	chunkTerms := termCounts(text)
	var score float64
	var matched int
	for term := range queryTerms {
		count := chunkTerms[term]
		if count == 0 {
			continue
		}
		matched++
		score += 1 + math.Log(float64(count))
	}
	if matched == 0 {
		return 0
	}
	score += float64(matched) / float64(len(queryTerms))
	if strings.Contains(strings.ToLower(text), queryLower) {
		score += 1
	}
	return score
}

func tokenSet(text string) map[string]int {
	counts := termCounts(text)
	for term := range counts {
		if len(term) < 2 {
			delete(counts, term)
		}
	}
	return counts
}

func termCounts(text string) map[string]int {
	counts := make(map[string]int)
	for _, term := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if term == "" {
			continue
		}
		counts[term]++
	}
	return counts
}

func splitLines(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.TrimSuffix(text, "\n")
	if text == "" {
		return nil
	}
	return strings.Split(text, "\n")
}

func excerpt(text string) string {
	text = strings.TrimSpace(strings.ToValidUTF8(text, "\uFFFD"))
	if len(text) <= maxExcerptBytes {
		return text
	}
	return strings.TrimSpace(truncateUTF8(text, maxExcerptBytes)) + "\n..."
}

func truncateUTF8(text string, limit int) string {
	if limit <= 0 || len(text) <= limit {
		return text
	}
	for limit > 0 && !utf8.RuneStart(text[limit]) {
		limit--
	}
	return text[:limit]
}

func filepathSlash(path string) string {
	return strings.ReplaceAll(path, "\\", "/")
}

func cleanMemoryFilePath(relPath string) (string, error) {
	relPath = strings.TrimSpace(relPath)
	if relPath == "" {
		return "", errors.New("memory path is required")
	}
	clean := path.Clean(filepathSlash(relPath))
	if path.IsAbs(clean) || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", fmt.Errorf("invalid memory path %q", relPath)
	}
	if clean == RootFile {
		return clean, nil
	}
	if strings.HasPrefix(clean, DailyDir+"/") && strings.HasSuffix(strings.ToLower(clean), ".md") {
		return clean, nil
	}
	return "", fmt.Errorf("memory path must be %s or %s/*.md", RootFile, DailyDir)
}
