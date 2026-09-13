# Gap report

## Repository inspection

This workspace contained only `Colitech_Agentic_AI_Final_Prompt.txt`. There was no existing `src/aar` package, graph, tests, README, Colitech graph image, node-interaction narrative, or Adaptive Multi-Agent Research Assistant v2 runbook. The implementation is a greenfield local MVP under `src/aar` following the final prompt. Older Monk/AWS teaching material was not used as the architecture.

## Exercised vs unavailable

| Path | How it was exercised |
| --- | --- |
| Fake provider | Automated tests (`AAR_ENABLE_FAKE=true` in the test harness) |
| Labeled web fixtures | Tests and local UI (`AAR_WEB_MODE=fixture`) |
| Local document keyword index | Tests and documents-only API/UI runs |
| Ollama `llama3.1:8b` | Live generate smoke from this machine (`AAR_OLLAMA_CHAT_MODEL=llama3.1:8b`). Default pytest still skips the marked live test unless `AAR_LIVE_OLLAMA=true`. |
| Tavily live search | Not exercised (no `AAR_TAVILY_API_KEY`) |
| Live page fetch | Not exercised; SSRF checks are unit-tested |
| Gemini / Anthropic / OpenRouter / Bedrock | Adapters present; not called (no keys / Bedrock not enabled) |

## Remaining limitations

- Live paid models stay disabled until the matching environment key is set. Catalog vendor IDs and prices were reviewed against official docs on 2026-09-13 and should be re-checked before a paid call.
- Tavily billing is not in the hard request ceiling unless `AAR_TAVILY_COST_USD` is set.
- The browser UI was verified as HTML/API routes through TestClient, not a live desktop browser session.
- Vector search and a hosted database are intentionally out of scope.

## Optional later work

See `docs/FOLLOW_ON.md` for Bedrock guardrails, LangSmith, a JSONL eval set, and AWS hosting. Those items were not started.
