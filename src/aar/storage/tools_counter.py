from __future__ import annotations

import threading
from collections import defaultdict


class ToolUseCounter:
    """Cumulative tool uses per sub-question across workers and rounds."""

    def __init__(self, limit: int = 4) -> None:
        self.limit = limit
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def remaining(self, run_id: str, sub_question_id: str) -> int:
        with self._lock:
            used = self._counts[(run_id, sub_question_id)]
            return max(0, self.limit - used)

    def consume(self, run_id: str, sub_question_id: str, n: int = 1) -> bool:
        with self._lock:
            key = (run_id, sub_question_id)
            if self._counts[key] + n > self.limit:
                return False
            self._counts[key] += n
            return True

    def snapshot(self, run_id: str) -> dict[str, int]:
        with self._lock:
            return {
                sq_id: count
                for (rid, sq_id), count in self._counts.items()
                if rid == run_id
            }
