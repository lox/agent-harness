// Package memory provides an optional file-backed memory layer for agent
// applications built on agent-harness.
package memory

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	// RootFile is the curated long-term memory file in a workspace.
	RootFile = "MEMORY.md"

	// DailyDir is the directory that holds dated working memory files.
	DailyDir = "memory"
)

const defaultBootstrapLimit = 64 * 1024

// Store reads and writes memory files under a workspace directory.
//
// Markdown files are the durable source of truth. Search indexes or vector
// stores can be layered behind this API later without changing the file
// contract.
type Store struct {
	workspaceDir       string
	now                func() time.Time
	bootstrapByteLimit int
	mu                 sync.Mutex
}

// Option configures a Store.
type Option func(*Store)

// WithClock sets the clock used for dated memory files. It is primarily useful
// in tests.
func WithClock(now func() time.Time) Option {
	return func(s *Store) {
		if now != nil {
			s.now = now
		}
	}
}

// WithBootstrapByteLimit sets the maximum number of bytes loaded into the
// prompt section. Values <= 0 disable truncation.
func WithBootstrapByteLimit(limit int) Option {
	return func(s *Store) {
		s.bootstrapByteLimit = limit
	}
}

// New creates a memory store rooted at workspaceDir.
func New(workspaceDir string, opts ...Option) (*Store, error) {
	if strings.TrimSpace(workspaceDir) == "" {
		return nil, errors.New("memory workspace directory is required")
	}

	abs, err := filepath.Abs(workspaceDir)
	if err != nil {
		return nil, fmt.Errorf("resolve memory workspace: %w", err)
	}

	store := &Store{
		workspaceDir:       abs,
		now:                time.Now,
		bootstrapByteLimit: defaultBootstrapLimit,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(store)
		}
	}

	return store, nil
}

// WorkspaceDir returns the absolute workspace directory used by the store.
func (s *Store) WorkspaceDir() string {
	if s == nil {
		return ""
	}
	return s.workspaceDir
}

// Ensure creates the workspace directory and daily memory directory.
func (s *Store) Ensure(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s == nil {
		return errors.New("memory store is nil")
	}
	if err := os.MkdirAll(s.workspaceDir, 0o755); err != nil {
		return fmt.Errorf("create memory workspace: %w", err)
	}
	root, err := s.openRoot()
	if err != nil {
		return err
	}
	defer root.Close()
	if err := root.MkdirAll(DailyDir, 0o755); err != nil {
		return fmt.Errorf("create daily memory directory: %w", err)
	}
	return nil
}

// PromptSection returns the memory prompt section for the current turn. It
// includes MEMORY.md plus today's and yesterday's dated memory files when they
// exist.
func (s *Store) PromptSection(ctx context.Context) (string, error) {
	if err := s.Ensure(ctx); err != nil {
		return "", err
	}
	root, err := s.openRoot()
	if err != nil {
		return "", err
	}
	defer root.Close()

	refs, err := s.bootstrapRefs(root)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	b.WriteString("## Memory\n")
	b.WriteString("Memory is stored in Markdown files. Use memory_search before answering questions about prior work, preferences, decisions, plans, people, dates, or stored context, then use memory_get for exact file excerpts when needed.\n")
	if len(refs) == 0 {
		b.WriteString("\nNo memory files are currently loaded.\n")
		return b.String(), nil
	}

	remaining := s.bootstrapByteLimit
	truncated := false
	for _, ref := range refs {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		content, err := root.ReadFile(ref.relPath)
		if err != nil {
			return "", fmt.Errorf("read memory file %s: %w", ref.relPath, err)
		}
		text := strings.TrimSpace(strings.ToValidUTF8(string(content), "\uFFFD"))
		if text == "" {
			continue
		}
		if remaining > 0 && len(text) > remaining {
			text = truncateUTF8(text, remaining)
			truncated = true
		}

		b.WriteString("\n### ")
		b.WriteString(ref.relPath)
		b.WriteString("\n")
		b.WriteString(text)
		b.WriteString("\n")

		if remaining > 0 {
			remaining -= len(text)
			if remaining <= 0 {
				truncated = true
				break
			}
		}
	}
	if truncated {
		b.WriteString("\nMemory prompt content was truncated. Use memory_search and memory_get for additional context.\n")
	}
	return b.String(), nil
}

// Bootstrap appends the memory prompt section to a base system prompt.
func (s *Store) Bootstrap(ctx context.Context, baseSystem string) (string, error) {
	section, err := s.PromptSection(ctx)
	if err != nil {
		return "", err
	}
	baseSystem = strings.TrimSpace(baseSystem)
	if baseSystem == "" {
		return section, nil
	}
	return baseSystem + "\n\n" + section, nil
}

// AppendDaily appends a Markdown entry to today's working memory file and
// returns the relative path that was written.
func (s *Store) AppendDaily(ctx context.Context, title, body string) (string, error) {
	if err := s.Ensure(ctx); err != nil {
		return "", err
	}
	body = strings.TrimSpace(body)
	if body == "" {
		return "", errors.New("memory body cannot be empty")
	}

	title = strings.TrimSpace(title)
	if title == "" {
		title = "Memory"
	}

	now := s.now()
	relPath := path.Join(DailyDir, now.Format("2006-01-02")+".md")
	root, err := s.openRoot()
	if err != nil {
		return "", err
	}
	defer root.Close()

	var entry strings.Builder
	entry.WriteString("\n## ")
	entry.WriteString(now.Format("15:04:05"))
	entry.WriteString(" ")
	entry.WriteString(title)
	entry.WriteString("\n\n")
	entry.WriteString(body)
	entry.WriteString("\n")

	f, err := root.OpenFile(relPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", fmt.Errorf("open daily memory: %w", err)
	}
	defer f.Close()
	if _, err := f.WriteString(entry.String()); err != nil {
		return "", fmt.Errorf("write daily memory: %w", err)
	}
	return relPath, nil
}

type memoryRef struct {
	relPath string
}

func (s *Store) bootstrapRefs(root *os.Root) ([]memoryRef, error) {
	var refs []memoryRef

	rootRef, err := s.refFromPath(root, RootFile)
	if err != nil {
		return nil, err
	}
	if rootRef != nil {
		refs = append(refs, *rootRef)
	}

	now := s.now()
	for _, date := range []time.Time{now, now.AddDate(0, 0, -1)} {
		matches, err := s.dateRefs(root, date)
		if err != nil {
			return nil, err
		}
		refs = append(refs, matches...)
	}
	return refs, nil
}

func (s *Store) dateRefs(root *os.Root, date time.Time) ([]memoryRef, error) {
	pattern := path.Join(DailyDir, date.Format("2006-01-02")+"*.md")
	matches, err := fs.Glob(root.FS(), pattern)
	if err != nil {
		return nil, fmt.Errorf("glob dated memory files: %w", err)
	}
	sort.Strings(matches)

	refs := make([]memoryRef, 0, len(matches))
	for _, relPath := range matches {
		ref, err := s.refFromPath(root, relPath)
		if err != nil {
			return nil, err
		}
		if ref != nil {
			refs = append(refs, *ref)
		}
	}
	return refs, nil
}

func (s *Store) allMemoryRefs(root *os.Root) ([]memoryRef, error) {
	var refs []memoryRef

	rootRef, err := s.refFromPath(root, RootFile)
	if err != nil {
		return nil, err
	}
	if rootRef != nil {
		refs = append(refs, *rootRef)
	}

	if err := fs.WalkDir(root.FS(), DailyDir, func(relPath string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if errors.Is(walkErr, os.ErrNotExist) {
				return nil
			}
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		if strings.ToLower(path.Ext(entry.Name())) != ".md" {
			return nil
		}
		ref, err := s.refFromPath(root, relPath)
		if err != nil {
			return err
		}
		if ref != nil {
			refs = append(refs, *ref)
		}
		return nil
	}); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("walk memory files: %w", err)
	}

	sort.SliceStable(refs, func(i, j int) bool {
		if refs[i].relPath == refs[j].relPath {
			return false
		}
		if refs[i].relPath == RootFile {
			return true
		}
		if refs[j].relPath == RootFile {
			return false
		}
		return refs[i].relPath < refs[j].relPath
	})
	return refs, nil
}

func (s *Store) refFromPath(root *os.Root, relPath string) (*memoryRef, error) {
	info, err := root.Lstat(relPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("stat memory file: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, nil
	}
	return &memoryRef{relPath: filepath.ToSlash(relPath)}, nil
}

func (s *Store) openRoot() (*os.Root, error) {
	if s == nil {
		return nil, errors.New("memory store is nil")
	}
	root, err := os.OpenRoot(s.workspaceDir)
	if err != nil {
		return nil, fmt.Errorf("open memory workspace: %w", err)
	}
	return root, nil
}
