# Web Search Workflow

## Objective
Retrieve current external information and summarize it for coding context.

## Required Inputs
- Search query
- Preferred sources
- Max result count

## Agents Used
- Web Search Agent
- Memory Agent

## Tools Used
- web_search

## Expected Outputs
- Normalized search results
- Short summary
- Source URLs where available

## Edge Cases
- Provider API keys missing
- Conflicting search results
- Documentation pages that require source priority

## Validation Checks
- Prefer official docs for technical facts
- Mark fallback/no-key results clearly
