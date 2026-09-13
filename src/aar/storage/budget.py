"""Atomic per-run budget reservations shared by parallel workers."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict

from aar.errors import BudgetRejected


class BudgetLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent: dict[str, float] = defaultdict(float)
        self._reserved: dict[str, float] = defaultdict(float)
        self._reservations: dict[str, tuple[str, float]] = {}
        self._paid_calls: dict[str, int] = defaultdict(int)

    def snapshot(self, run_id: str) -> tuple[float, float]:
        with self._lock:
            return self._spent[run_id], self._reserved[run_id]

    def remaining(self, run_id: str, ceiling: float) -> float:
        spent, reserved = self.snapshot(run_id)
        return max(0.0, ceiling - spent - reserved)

    def paid_calls(self, run_id: str) -> int:
        with self._lock:
            return self._paid_calls[run_id]

    def reserve(self, run_id: str, ceiling: float, amount: float, paid: bool = True) -> str:
        if amount < 0:
            raise BudgetRejected("Reservation amount cannot be negative.")
        with self._lock:
            if amount == 0:
                reservation_id = str(uuid.uuid4())
                self._reservations[reservation_id] = (run_id, 0.0)
                return reservation_id
            projected = self._spent[run_id] + self._reserved[run_id] + amount
            if projected > ceiling + 1e-12:
                raise BudgetRejected(
                    "Remaining budget cannot cover the reserved worst-case cost."
                )
            reservation_id = str(uuid.uuid4())
            self._reserved[run_id] += amount
            self._reservations[reservation_id] = (run_id, amount)
            if paid:
                self._paid_calls[run_id] += 1
            return reservation_id

    def settle(self, reservation_id: str, actual: float) -> None:
        with self._lock:
            if reservation_id not in self._reservations:
                return
            run_id, reserved = self._reservations.pop(reservation_id)
            self._reserved[run_id] = max(0.0, self._reserved[run_id] - reserved)
            charged = min(max(0.0, actual), reserved) if reserved > 0 else 0.0
            if reserved == 0:
                charged = 0.0
            else:
                charged = min(max(0.0, actual), reserved)
            self._spent[run_id] += charged

    def release(self, reservation_id: str) -> None:
        self.settle(reservation_id, 0.0)
