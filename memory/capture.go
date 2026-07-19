package memory

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"strings"

	harness "github.com/lox/agent-harness"
)

const defaultCaptureMessages = 15

// CaptureOptions controls how recent conversation context is written to memory.
type CaptureOptions struct {
	// MaxMessages limits how many trailing messages are captured. Values <= 0
	// use a conservative default.
	MaxMessages int

	// Title becomes the top-level heading in the captured memory file.
	Title string

	// Slug is appended to the timestamped filename after sanitization.
	Slug string

	// IncludeToolResults controls whether tool result content is captured.
	IncludeToolResults bool
}

// CaptureThread writes recent thread context into a timestamped file under
// memory/. This is useful for /new, /reset, or pre-compaction flush flows.
func (s *Store) CaptureThread(ctx context.Context, thread *harness.Thread, opts CaptureOptions) (string, error) {
	if thread == nil {
		return "", errors.New("thread is nil")
	}
	return s.CaptureMessages(ctx, thread.Messages, opts)
}

// CaptureMessages writes recent messages into a timestamped memory file and
// returns the relative path.
func (s *Store) CaptureMessages(ctx context.Context, messages []harness.Message, opts CaptureOptions) (string, error) {
	if err := s.Ensure(ctx); err != nil {
		return "", err
	}
	if len(messages) == 0 {
		return "", errors.New("no messages to capture")
	}
	if opts.MaxMessages <= 0 {
		opts.MaxMessages = defaultCaptureMessages
	}
	if len(messages) > opts.MaxMessages {
		messages = messages[len(messages)-opts.MaxMessages:]
	}

	now := s.now()
	slug := sanitizeSlug(opts.Slug)
	name := now.Format("2006-01-02-150405")
	if slug != "" {
		name += "-" + slug
	}
	name += ".md"

	root, err := s.openRoot()
	if err != nil {
		return "", err
	}
	defer root.Close()
	relPath, f, err := s.createCaptureFile(root, name)
	if err != nil {
		return "", err
	}
	defer f.Close()

	title := strings.TrimSpace(opts.Title)
	if title == "" {
		title = "Session Memory"
	}

	var b strings.Builder
	b.WriteString("# ")
	b.WriteString(title)
	b.WriteString("\n\n")
	b.WriteString("Captured: ")
	b.WriteString(now.Format("2006-01-02 15:04:05 MST"))
	b.WriteString("\n\n")
	b.WriteString("## Recent Messages\n")

	for _, msg := range messages {
		writeMessage(&b, msg, opts.IncludeToolResults)
	}

	if _, err := f.WriteString(b.String()); err != nil {
		return "", fmt.Errorf("write captured memory: %w", err)
	}
	if err := f.Close(); err != nil {
		return "", fmt.Errorf("close captured memory: %w", err)
	}
	return relPath, nil
}

func (s *Store) createCaptureFile(root *os.Root, name string) (string, *os.File, error) {
	for i := 0; ; i++ {
		candidate := name
		if i > 0 {
			ext := path.Ext(name)
			base := strings.TrimSuffix(name, ext)
			candidate = fmt.Sprintf("%s-%d%s", base, i+1, ext)
		}
		relPath := path.Join(DailyDir, candidate)
		f, err := root.OpenFile(relPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			return relPath, f, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return "", nil, fmt.Errorf("create captured memory: %w", err)
		}
	}
}

func writeMessage(b *strings.Builder, msg harness.Message, includeToolResults bool) {
	b.WriteString("\n### ")
	b.WriteString(string(msg.Role))
	b.WriteString("\n\n")

	switch {
	case msg.ToolResult != nil:
		if !includeToolResults {
			b.WriteString("[tool result omitted]\n")
			return
		}
		b.WriteString(strings.TrimSpace(msg.ToolResult.Content))
		b.WriteString("\n")
	case len(msg.ToolCalls) > 0:
		b.WriteString("[assistant requested tool calls: ")
		for i, call := range msg.ToolCalls {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(call.Name)
		}
		b.WriteString("]\n")
	case strings.TrimSpace(msg.Content) != "":
		b.WriteString(strings.TrimSpace(msg.Content))
		b.WriteString("\n")
	default:
		b.WriteString("[empty message]\n")
	}
}

func sanitizeSlug(slug string) string {
	slug = strings.ToLower(strings.TrimSpace(slug))
	if slug == "" {
		return ""
	}
	var b strings.Builder
	lastDash := false
	for _, r := range slug {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		case r == '-' || r == '_' || r == ' ':
			if !lastDash && b.Len() > 0 {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}
