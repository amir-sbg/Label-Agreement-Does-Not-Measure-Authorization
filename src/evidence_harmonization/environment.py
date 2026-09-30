from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    case_id: str
    budget: int = 6
    steps: list[dict[str, Any]] = field(default_factory=list)
    terminal: dict[str, Any] | None = None

    @property
    def calls_used(self) -> int:
        return sum(1 for step in self.steps if step["type"] == "tool")

    @property
    def calls_remaining(self) -> int:
        return self.budget - self.calls_used


class CaseEnvironment:
    def __init__(self, case: dict[str, Any], budget: int = 6):
        self.case = case
        self.trajectory = Trajectory(case["case_id"], budget)

    def observation(self) -> dict[str, Any]:
        return {
            "case_id": self.case["case_id"],
            "study": self.case["study"],
            "task": self.case["task"],
            "proposal": self.case["initial_observation"],
            "available_tools": sorted(self.case["tools"]),
            "calls_remaining": self.trajectory.calls_remaining,
        }

    def call(self, tool_name: str) -> dict[str, Any]:
        if self.trajectory.terminal is not None:
            raise RuntimeError("Trajectory already terminated")
        if self.trajectory.calls_remaining <= 0:
            raise RuntimeError("Tool budget exhausted")
        if tool_name not in self.case["tools"]:
            result = {
                "status": "unavailable",
                "tool_name": tool_name,
                "reason": "tool or decisive evidence is unavailable for this case",
            }
        else:
            result = {
                "status": "ok",
                "tool_name": tool_name,
                "evidence": self.case["tools"][tool_name],
            }
        self.trajectory.steps.append(
            {"type": "tool", "tool_name": tool_name, "result": result}
        )
        return result

    def terminate(
        self,
        action: str,
        label: str | None,
        repair: dict[str, Any] | None,
        evidence_tools: list[str],
        unresolved: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.trajectory.terminal is not None:
            raise RuntimeError("Trajectory already terminated")
        terminal = {
            "action": action,
            "label": label,
            "repair": repair,
            "evidence_tools": evidence_tools,
            "unresolved": unresolved or [],
        }
        self.trajectory.terminal = terminal
        self.trajectory.steps.append({"type": "terminal", "decision": terminal})
        return terminal
