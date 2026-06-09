# Sandbox 02: Conditional Edges with Feature Flags

## Status: INCOMPLETE — needs more investigation

## Finding
ADK Workflow's conditional routing via `(node, {'value': target_node})` edge
syntax compiles but doesn't route correctly in my tests. The FunctionNode
executes and returns a value, but the downstream agent doesn't receive the
conversation context.

## What works
- FunctionNode with `parameter_binding='state'` reads session state
- Routing map syntax `(flag_node, {'v1': v1_agent, 'v2': v2_agent})` is accepted
- Workflow builds without errors

## What doesn't work
- The routed-to agent doesn't produce output (1 event with no text)
- May need to check ADK's routing API more carefully
- The function return value may need specific typing or edge handling

## Next steps
- Study ADK docs for conditional routing examples
- Check if `RouteValue` type constraints apply
- Try with explicit Edge objects instead of tuples
