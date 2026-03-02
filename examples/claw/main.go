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

func main() {
	modelFlag := flag.String("model", envOrDefault("OPENAI_MODEL", "gpt-4o-mini"), "model name")
	systemFlag := flag.String("system", "You are Claw, a concise command-line coding assistant.", "system prompt")
	maxStepsFlag := flag.Int("max-steps", 8, "max tool loop steps per turn")
	flag.Parse()

	provider := openai.New(openai.WithDefaultModel(*modelFlag))
	r := runner.New()
	thread := harness.NewThread()
	tools := builtInTools()

	var mu sync.Mutex

	fmt.Println("claw repl")
	fmt.Println("commands: /help, /stop, /history, /tools, /quit")

	input := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("claw> ")
		if !input.Scan() {
			fmt.Println()
			break
		}

		line := strings.TrimSpace(input.Text())
		if line == "" {
			continue
		}

		switch line {
		case "/help":
			fmt.Println("/help       show commands")
			fmt.Println("/stop       cancel active run")
			fmt.Println("/history    print thread messages")
			fmt.Println("/tools      list available tools")
			fmt.Println("/quit       exit")
			continue
		case "/tools":
			for _, tool := range tools {
				fmt.Printf("- %s: %s\n", tool.Name, tool.Description)
			}
			continue
		case "/history":
			mu.Lock()
			for i, msg := range thread.Messages {
				fmt.Printf("%d. %s", i+1, msg.Role)
				if msg.Content != "" {
					fmt.Printf(": %s", msg.Content)
				}
				if msg.ToolResult != nil {
					fmt.Printf(": %s", msg.ToolResult.Content)
				}
				fmt.Println()
			}
			mu.Unlock()
			continue
		case "/stop":
			if r.Stop(thread.ID) {
				fmt.Println("active run cancelled")
			} else {
				fmt.Println("no active run")
			}
			continue
		case "/quit":
			r.Stop(thread.ID)
			return
		}

		if r.IsRunning(thread.ID) {
			fmt.Println("run already active; use /stop first")
			continue
		}

		mu.Lock()
		thread.AddUser(line)
		messages := append([]harness.Message(nil), thread.Messages...)
		mu.Unlock()

		done, err := r.Start(context.Background(), thread.ID, func(ctx context.Context) error {
			result, err := harness.Run(ctx, provider,
				harness.WithSystem(*systemFlag),
				harness.WithMessages(messages...),
				harness.WithTools(tools...),
				harness.WithModel(*modelFlag),
				harness.WithMaxSteps(*maxStepsFlag),
			)
			if err != nil {
				return err
			}

			mu.Lock()
			thread.Append(result)
			mu.Unlock()

			for _, msg := range result.Messages {
				switch msg.Role {
				case harness.RoleAssistant:
					if msg.Content != "" {
						fmt.Printf("assistant> %s\n", msg.Content)
					}
				case harness.RoleTool:
					if msg.ToolResult != nil && msg.ToolResult.UserContent != "" {
						fmt.Printf("tool[%s]> %s\n", msg.ToolResult.ToolCallID, msg.ToolResult.UserContent)
					}
				}
			}

			if result.StopReason == harness.StopMaxSteps {
				fmt.Println("assistant> stopped due to max steps")
			}
			return nil
		})
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}

		go func() {
			err := <-done
			switch {
			case err == nil:
			case errors.Is(err, context.Canceled):
				fmt.Println("assistant> cancelled")
			default:
				fmt.Printf("assistant error: %v\n", err)
			}
		}()
	}

	if err := input.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "read error: %v\n", err)
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
