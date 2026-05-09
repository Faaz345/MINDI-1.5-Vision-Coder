# Generate UI Workflow

## Objective
Create or update user-facing UI code from a structured product prompt.

## Required Inputs
- User prompt
- Active files when available
- Design settings and project constraints

## Agents Used
- Intent Agent
- Planner Agent
- Workflow Router
- Validation Agent
- Memory Agent

## Tools Used
- analyze_repo
- write_files
- lint_project

## Expected Outputs
- File deltas for generated or edited UI files
- Concise implementation summary
- Validation warnings or next checks

## Edge Cases
- Vague prompts require clarification
- Missing framework context should default to the active repo stack
- Generated code must avoid placeholder-only output

## Validation Checks
- Referenced files exist
- Build/lint checks run when available
- UI remains responsive and accessible
