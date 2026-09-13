# Optional follow-on work

These items come from the older Monk teaching exercise. They are not part of the local Colitech MVP and were not implemented here.

1. Bedrock guardrail demo
   Add an optional safety demonstration that sends a labeled prompt through a Bedrock guardrail and records allow/block outcomes. Keep the multi-provider catalog intact. Do not hard-code a service name or a single Bedrock model.

2. LangSmith traces
   Emit traces for planner, workers, writer, and QA when a LangSmith key is present. Keep default telemetry redacted and local.

3. JSONL evaluation set
   Build a small evaluation file with deterministic citation checks plus planner/report graders. Reuse the existing validator and Final QA rules rather than introducing a second evidence policy.

4. AWS deployment
   If the product later needs hosting, choose compute and storage from current AWS documentation. Do not carry forward RDS/pgvector, ECS, or Monk-specific names as hidden requirements of this local SQLite MVP.

Do not provision paid cloud resources from this repository unless a later prompt explicitly asks for that work.
