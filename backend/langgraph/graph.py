from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Node:
    id: str
    agent_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    validator: Optional["Validator"] = None


@dataclass
class Graph:
    id: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node


def to_dict(g: Graph) -> dict:
    return {
        "id": g.id,
        "created_at": g.created_at,
        "nodes": [
            {
                "id": n.id,
                "agent_type": n.agent_type,
                "inputs": n.inputs,
                "outputs": n.outputs,
                "status": n.status,
                "timestamp": n.timestamp,
            }
            for n in g.nodes
        ],
        "edges": [{"from": a, "to": b} for (a, b) in g.edges],
    }
