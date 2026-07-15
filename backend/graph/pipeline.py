"""Graph assembly.

Topology (fixed — conditional edges are deterministic functions of state):

    input_validation ──(rejected)──> END
          │
          v
     route_query ──(small talk)──> small_talk ──> END
          │
    (document question)
          v
       retrieve ──> grade ──(not relevant)──> refuse ──> END
                      │
                (relevant)
                      v
                  generate ──> groundedness_check ──(grounded)──> END
                      ^                │
                      │        (not grounded, first time)
                      └────────────────┘
                               │
                     (not grounded, after retry)
                               v
                            refuse ──> END

The sixth stage from the design ("stream_out") is realized as the SSE layer in
main.py, which consumes this graph's event stream — streaming is transport,
not a reasoning step, so it lives outside the graph.
"""
from langgraph.graph import END, StateGraph

from graph.nodes import (
    generate_node,
    grade_node,
    groundedness_check_node,
    input_validation_node,
    refuse_node,
    retrieve_node,
    route_after_grade,
    route_after_groundedness,
    route_after_routing,
    route_after_validation,
    route_query_node,
    small_talk_node,
)
from graph.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("input_validation", input_validation_node)
    graph.add_node("route_query", route_query_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("groundedness_check", groundedness_check_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("input_validation")
    graph.add_conditional_edges(
        "input_validation",
        route_after_validation,
        {"route_query": "route_query", END: END},
    )
    graph.add_conditional_edges(
        "route_query",
        route_after_routing,
        {"small_talk": "small_talk", "retrieve": "retrieve"},
    )
    graph.add_edge("small_talk", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"generate": "generate", "refuse": "refuse"}
    )
    graph.add_edge("generate", "groundedness_check")
    graph.add_conditional_edges(
        "groundedness_check",
        route_after_groundedness,
        {END: END, "generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("refuse", END)

    return graph.compile()


pipeline = build_graph()
