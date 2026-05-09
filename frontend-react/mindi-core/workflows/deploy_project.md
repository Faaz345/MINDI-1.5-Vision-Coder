# Deploy Project Workflow

## Objective
Prepare a project for production deployment.

## Required Inputs
- Deployment target
- Environment requirements
- Active project files

## Agents Used
- Intent Agent
- Planner Agent
- Validation Agent

## Tools Used
- run_build
- lint_project

## Expected Outputs
- Build status
- Deployment checklist
- Required environment variables

## Edge Cases
- Missing secrets
- Build command not configured
- Provider-specific settings

## Validation Checks
- Production build succeeds
- Required env vars are documented
