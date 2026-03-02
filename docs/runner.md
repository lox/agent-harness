# Runner Helper

`runner` is an optional convenience package for managing in-flight harness runs.

It is useful when you need to process control-plane inputs (for example "stop") while a run is active.

## API

- `runner.New() *Runner`
- `(*Runner).Start(parent, runID, fn) (<-chan error, error)`
- `(*Runner).Stop(runID) bool`
- `(*Runner).IsRunning(runID) bool`

## Behaviour

- `Start` creates a cancellable context and executes `fn` in a goroutine
- `Start` returns a `done` channel that receives the function error when complete
- duplicate `runID` starts return `ErrAlreadyRunning`
- `Stop` calls the active run's cancel function and returns whether a run was found
- entries are cleaned up automatically when the run exits

## Example

```go
r := runner.New()

done, err := r.Start(ctx, threadID, func(runCtx context.Context) error {
    _, err := harness.Run(runCtx, provider, harness.WithMessages(msgs...))
    return err
})
if err != nil {
    return err
}

if incomingControl == "stop" {
    r.Stop(threadID)
}

if err := <-done; err != nil {
    // handle completion error
}
```
