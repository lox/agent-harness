package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	harness "github.com/lox/agent-harness"
)

var memorySearchSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The memory search query."
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of results to return.",
      "minimum": 1,
      "maximum": 20
    },
    "min_score": {
      "type": "number",
      "description": "Minimum lexical score to include."
    }
  },
  "required": ["query"],
  "additionalProperties": false
}`)

var memoryGetSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Relative memory file path, such as MEMORY.md or memory/2026-05-23.md."
    },
    "start_line": {
      "type": "integer",
      "description": "1-indexed first line to read. Omit or set 0 to start at the beginning.",
      "minimum": 0
    },
    "end_line": {
      "type": "integer",
      "description": "1-indexed final line to read. Omit or set 0 to read through the end.",
      "minimum": 0
    }
  },
  "required": ["path"],
  "additionalProperties": false
}`)

// Tools returns memory_search and memory_get tools for use with harness.Run.
func (s *Store) Tools() []harness.Tool {
	return []harness.Tool{s.SearchTool(), s.GetTool()}
}

// SearchTool returns a tool that searches MEMORY.md and memory/*.md.
func (s *Store) SearchTool() harness.Tool {
	return harness.Tool{
		ToolDef: harness.ToolDef{
			Name:        "memory_search",
			Description: "Search durable Markdown memory files for prior work, preferences, decisions, plans, people, dates, or stored context.",
			Parameters:  memorySearchSchema,
		},
		Execute: func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
			var args struct {
				Query      string  `json:"query"`
				MaxResults int     `json:"max_results"`
				MinScore   float64 `json:"min_score"`
			}
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return nil, fmt.Errorf("parse memory_search arguments: %w", err)
			}

			results, err := s.Search(ctx, args.Query, SearchOptions{
				MaxResults: args.MaxResults,
				MinScore:   args.MinScore,
			})
			if err != nil {
				return nil, err
			}
			recallRecorded := true
			recallErr := ""
			if err := s.RecordSearchResults(ctx, args.Query, results); err != nil {
				recallRecorded = false
				recallErr = err.Error()
			}

			payload := struct {
				Results        []SearchResult `json:"results"`
				Provider       string         `json:"provider"`
				Fallback       bool           `json:"fallback"`
				RecallRecorded bool           `json:"recall_recorded"`
				RecallError    string         `json:"recall_error,omitempty"`
			}{
				Results:        results,
				Provider:       "builtin-lexical",
				Fallback:       true,
				RecallRecorded: recallRecorded,
				RecallError:    recallErr,
			}
			content, err := json.MarshalIndent(payload, "", "  ")
			if err != nil {
				return nil, fmt.Errorf("encode memory_search result: %w", err)
			}
			return &harness.ToolResult{
				ToolCallID:  call.ID,
				Content:     string(content),
				UserContent: fmt.Sprintf("memory_search returned %d result(s)", len(results)),
			}, nil
		},
	}
}

// GetTool returns a tool that reads exact line ranges from memory files.
func (s *Store) GetTool() harness.Tool {
	return harness.Tool{
		ToolDef: harness.ToolDef{
			Name:        "memory_get",
			Description: "Read an exact line range from a durable Markdown memory file returned by memory_search.",
			Parameters:  memoryGetSchema,
		},
		Execute: func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
			var args struct {
				Path      string `json:"path"`
				StartLine int    `json:"start_line"`
				EndLine   int    `json:"end_line"`
			}
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return nil, fmt.Errorf("parse memory_get arguments: %w", err)
			}
			if args.Path == "" {
				return nil, errors.New("memory_get path is required")
			}

			result, err := s.Get(ctx, args.Path, args.StartLine, args.EndLine)
			if err != nil {
				return nil, err
			}
			content, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				return nil, fmt.Errorf("encode memory_get result: %w", err)
			}
			return &harness.ToolResult{
				ToolCallID:  call.ID,
				Content:     string(content),
				UserContent: fmt.Sprintf("memory_get read %s:%d", result.Path, result.StartLine),
			}, nil
		},
	}
}
