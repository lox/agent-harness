package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	harness "github.com/lox/agent-harness"
	openai "github.com/lox/agent-harness/provider/openai"
	"github.com/lox/agent-harness/runner"
)

const (
	prompt = "claw> "
)

type app struct {
	provider harness.Provider
	runner   *runner.Runner
	thread   *harness.Thread
	tools    []harness.Tool

	system   string
	model    string
	maxSteps int

	mu sync.Mutex
}

func main() {
	modelFlag := flag.String("model", envOrDefault("OPENAI_MODEL", "gpt-4o-mini"), "model name")
	systemFlag := flag.String("system", "You are Claw, a concise command-line coding assistant.", "system prompt")
	maxStepsFlag := flag.Int("max-steps", 8, "max tool loop steps per turn")
	flag.Parse()

	a := &app{
		provider: openai.New(openai.WithDefaultModel(*modelFlag)),
		runner:   runner.New(),
		thread:   harness.NewThread(),
		tools:    builtInTools(),
		system:   *systemFlag,
		model:    *modelFlag,
		maxSteps: *maxStepsFlag,
	}
	a.run()
}

func (a *app) run() {
	fmt.Println("claw repl")
	fmt.Println("commands: /help, /stop, /history, /tools, /quit")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(prompt)
		if !scanner.Scan() {
			fmt.Println()
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		handled, quit := a.handleCommand(line)
		if quit {
			return
		}
		if handled {
			continue
		}

		a.handlePrompt(line)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "read error: %v\n", err)
	}
}

func (a *app) handleCommand(line string) (handled bool, quit bool) {
	switch line {
	case "/help":
		fmt.Println("/help       show commands")
		fmt.Println("/stop       cancel active run")
		fmt.Println("/history    print thread messages")
		fmt.Println("/tools      list available tools")
		fmt.Println("/quit       exit")
		return true, false
	case "/tools":
		for _, tool := range a.tools {
			fmt.Printf("- %s: %s\n", tool.Name, tool.Description)
		}
		return true, false
	case "/history":
		a.printHistory()
		return true, false
	case "/stop":
		if a.runner.Stop(a.thread.ID) {
			fmt.Println("assistant │ cancelled active run")
		} else {
			fmt.Println("assistant │ no active run")
		}
		return true, false
	case "/quit":
		a.runner.Stop(a.thread.ID)
		return true, true
	default:
		return false, false
	}
}

func (a *app) handlePrompt(text string) {
	if a.runner.IsRunning(a.thread.ID) {
		fmt.Println("assistant │ run already active; use /stop first")
		return
	}

	a.mu.Lock()
	a.thread.AddUser(text)
	messages := append([]harness.Message(nil), a.thread.Messages...)
	a.mu.Unlock()

	done, err := a.runner.Start(context.Background(), a.thread.ID, func(ctx context.Context) error {
		result, err := harness.Run(ctx, a.provider,
			harness.WithSystem(a.system),
			harness.WithMessages(messages...),
			harness.WithTools(a.tools...),
			harness.WithModel(a.model),
			harness.WithMaxSteps(a.maxSteps),
		)
		if err != nil {
			return err
		}

		a.mu.Lock()
		a.thread.Append(result)
		a.mu.Unlock()

		a.printResult(result)
		return nil
	})
	if err != nil {
		fmt.Printf("assistant │ error: %v\n", err)
		return
	}

	go func() {
		err := <-done
		switch {
		case err == nil:
		case errors.Is(err, context.Canceled):
			fmt.Println("assistant │ cancelled")
		default:
			fmt.Printf("assistant │ error: %v\n", err)
		}
	}()
}

func (a *app) printHistory() {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.thread.Messages) == 0 {
		fmt.Println("history │ empty")
		return
	}

	for i, msg := range a.thread.Messages {
		content := strings.TrimSpace(msg.Content)
		if msg.ToolResult != nil {
			content = strings.TrimSpace(msg.ToolResult.Content)
		}
		fmt.Printf("%02d │ %-9s │ %s\n", i+1, msg.Role, content)
	}
}

func (a *app) printResult(result *harness.Result) {
	var lastToolOutput string

	for _, msg := range result.Messages {
		switch msg.Role {
		case harness.RoleTool:
			if msg.ToolResult == nil {
				continue
			}
			out := msg.ToolResult.UserContent
			if out == "" {
				out = msg.ToolResult.Content
			}
			out = strings.TrimSpace(out)
			if out == "" {
				continue
			}
			fmt.Printf("tool      │ %s\n", out)
			lastToolOutput = out
		case harness.RoleAssistant:
			content := strings.TrimSpace(msg.Content)
			if content == "" || content == lastToolOutput {
				continue
			}
			fmt.Printf("assistant │ %s\n", content)
			lastToolOutput = ""
		}
	}

	if result.StopReason == harness.StopMaxSteps {
		fmt.Println("assistant │ stopped due to max steps")
	}
}

func builtInTools() []harness.Tool {
	return []harness.Tool{
		{
			ToolDef: harness.ToolDef{
				Name:        "echo",
				Description: "Echo back a string",
				Parameters: json.RawMessage(`{
				  "type":"object",
				  "properties":{"text":{"type":"string"}},
				  "required":["text"]
				}`),
			},
			Execute: func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
				var args struct {
					Text string `json:"text"`
				}
				_ = json.Unmarshal(call.Arguments, &args)
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				default:
				}
				return &harness.ToolResult{
					ToolCallID:  call.ID,
					Content:     args.Text,
					UserContent: args.Text,
				}, nil
			},
		},
		{
			ToolDef: harness.ToolDef{
				Name:        "time_now",
				Description: "Get the current local time",
				Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
			},
			Execute: func(ctx context.Context, call harness.ToolCall) (*harness.ToolResult, error) {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				default:
				}
				now := time.Now().Format(time.RFC3339)
				return &harness.ToolResult{
					ToolCallID:  call.ID,
					Content:     now,
					UserContent: now,
				}, nil
			},
		},
	}
}

func envOrDefault(key, fallback string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	return v
}
