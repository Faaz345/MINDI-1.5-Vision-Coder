# Debug Error Workflow

## Objective
Diagnose and fix a reported bug or runtime/build failure.

## Required Inputs
- Error message or reproduction steps
- Relevant files or stack trace

## Agents Used
- Intent Agent
- Planner Agent
- Validation Agent
- Memory Agent

## Tools Used
- read_files
- run_build
- lint_project

## Expected Outputs
- Root-cause summary
- Targeted code edits
- Verification result

## Edge Cases
- Missing stack traces
- Intermittent browser-only failures
- Dependency or environment mismatch

## Validation Checks
- Re-run failing command where possible
- Keep fix scoped to the failing behavior
