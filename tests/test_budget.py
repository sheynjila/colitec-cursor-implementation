from __future__ import annotations

import threading

import pytest

from aar.errors import BudgetRejected
from aar.storage.budget import BudgetLedger


def test_zero_budget_rejects_paid_reservation() -> None:
    ledger = BudgetLedger()
    with pytest.raises(BudgetRejected):
        ledger.reserve("run-zero", ceiling=0.0, amount=0.01, paid=True)
    reservation = ledger.reserve("run-zero", ceiling=0.0, amount=0.0, paid=False)
    ledger.settle(reservation, 0.0)
    assert ledger.paid_calls("run-zero") == 0
    spent, reserved = ledger.snapshot("run-zero")
    assert spent == 0
    assert reserved == 0


def test_concurrent_workers_cannot_overspend() -> None:
    ledger = BudgetLedger()
    results: list[str] = []

    def worker() -> None:
        try:
            reservation = ledger.reserve("run-c", ceiling=1.0, amount=0.6, paid=True)
            ledger.settle(reservation, 0.6)
            results.append("ok")
        except BudgetRejected:
            results.append("rejected")

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("ok") == 1 or results.count("ok") == 2
    spent, reserved = ledger.snapshot("run-c")
    assert spent + reserved <= 1.0 + 1e-9
    assert results.count("rejected") >= 1
