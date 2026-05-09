# Refactor Code Workflow

## Objective
Improve code structure without changing user-visible behavior.

## Required Inputs
- Refactor target
- Existing files and behavior constraints

## Agents Used
- Intent Agent
- Planner Agent
- Validation Agent

## Tools Used
- read_files
- write_files
- lint_project

## Expected Outputs
- Focused file edits
- Behavior preservation notes
- Validation output

## Edge Cases
- Ambiguous ownership boundaries
- Unrelated dirty worktree changes
- Refactors that expand scope unexpectedly

## Validation Checks
- Build/lint passes where available
- No unrelated rewrites
