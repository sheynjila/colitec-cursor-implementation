# Colitech graph

```mermaid
flowchart TD
    API[API Gateway] --> REG[Run Registry]
    REG --> SVC[Run Service]
    SVC --> START((START))
    START --> IN[Request Intake]
    IN --> PL[Planner]
    PL --> CR[Complexity Router]
    CR --> RS[Research Supervisor]
    RS -->|Send web| WW[Web Search Worker]
    RS -->|Send documents| DW[Document Search Worker]
    WW --> RJ[Research Join]
    DW --> RJ
    RJ --> VA[Validator]
    VA --> CG[Coverage Gate]
    CG -->|retry uncovered only| RS
    CG --> WR[Writer]
    WR --> QA[Final QA]
    QA --> LU[Learning Update]
    LU --> END((END))
```

API Gateway, Run Registry, and Run Service stay outside LangGraph. The compiled graph is created once and reused. The summarizer is a worker utility, not a dispatched node. User feedback is stored after the run through the API.
