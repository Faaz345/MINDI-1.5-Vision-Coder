# Analyze Repo Workflow

## Objective
Understand project structure, stack, and relevant files before planning code changes.

## Required Inputs
- Repository root
- Optional target path or user question

## Agents Used
- Intent Agent
- Planner Agent
- Memory Agent

## Tools Used
- analyze_repo
- read_files

## Expected Outputs
- Detected stack
- Important files
- Suggested next action

## Edge Cases
- Large repositories
- Generated directories
- Missing package metadata

## Validation Checks
- Ignore build artifacts and dependencies
- Keep analysis bounded
