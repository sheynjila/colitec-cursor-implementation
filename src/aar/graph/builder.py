from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Any

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from aar.graph.state import GraphState
from aar.nodes.complexity_router import complexity_router
from aar.nodes.coverage_gate import coverage_gate, route_after_coverage
from aar.nodes.document_search_worker import document_search_worker
from aar.nodes.final_qa import final_qa
from aar.nodes.learning_update import learning_update
from aar.nodes.planner import planner
from aar.nodes.request_intake import request_intake
from aar.nodes.research_join import research_join
from aar.nodes.research_supervisor import dispatch_workers, research_supervisor
from aar.nodes.validator import validator
from aar.nodes.web_search_worker import web_search_worker
from aar.nodes.writer import writer


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("request_intake", request_intake)
    graph.add_node("planner", planner)
    graph.add_node("complexity_router", complexity_router)
    graph.add_node("research_supervisor", research_supervisor)
    graph.add_node("web_search_worker", web_search_worker)
    graph.add_node("document_search_worker", document_search_worker)
    graph.add_node("research_join", research_join)
    graph.add_node("validator", validator)
    graph.add_node("coverage_gate", coverage_gate)
    graph.add_node("writer", writer)
    graph.add_node("final_qa", final_qa)
    graph.add_node("learning_update", learning_update)

    graph.add_edge(START, "request_intake")
    graph.add_edge("request_intake", "planner")
    graph.add_edge("planner", "complexity_router")
    graph.add_edge("complexity_router", "research_supervisor")
    graph.add_conditional_edges(
        "research_supervisor",
        dispatch_workers,
        ["web_search_worker", "document_search_worker", "research_join"],
    )
    graph.add_edge("web_search_worker", "research_join")
    graph.add_edge("document_search_worker", "research_join")
    graph.add_edge("research_join", "validator")
    graph.add_edge("validator", "coverage_gate")
    graph.add_conditional_edges(
        "coverage_gate",
        route_after_coverage,
        {
            "research_supervisor": "research_supervisor",
            "writer": "writer",
        },
    )
    graph.add_edge("writer", "final_qa")
    graph.add_edge("final_qa", "learning_update")
    graph.add_edge("learning_update", END)
    return graph


def compile_graph(checkpointer: Any | None = None) -> Any:
    return build_graph().compile(checkpointer=checkpointer or MemorySaver())


def sqlite_checkpointer(path: str) -> SqliteSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_compiled_graph(checkpoint_path: str) -> Any:
    return compile_graph(sqlite_checkpointer(checkpoint_path))
