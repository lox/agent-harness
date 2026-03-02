package harness

// EventType identifies an event emitted by the loop.
type EventType int

const (
	EventTurnStart EventType = iota // LLM call starting
	EventTurnEnd                    // LLM call completed
	EventToolStart                  // tool execution starting
	EventToolEnd                    // tool execution completed
	EventMessage                    // new message added to history
	EventError                      // non-fatal error (e.g. tool failure)
)

// Event represents a loop lifecycle event.
type Event struct {
	Type     EventType
	Step     int
	Message  *Message
	ToolCall *ToolCall
	Result   *ToolResult
	Error    error
}
