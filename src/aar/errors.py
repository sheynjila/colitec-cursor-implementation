"""Controlled error types that never leak secrets or stack traces to clients."""

from __future__ import annotations


class AARError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class ValidationError(AARError):
    def __init__(self, message: str, code: str = "invalid_request") -> None:
        super().__init__(code, message)


class ProviderError(AARError):
    def __init__(self, message: str, code: str = "provider_error") -> None:
        super().__init__(code, message)


class BudgetRejected(AARError):
    def __init__(self, message: str = "Request budget cannot fund this call.") -> None:
        super().__init__("budget_rejected", message)


class RouteUnavailable(AARError):
    def __init__(self, message: str = "No eligible model alias remains.") -> None:
        super().__init__("route_unavailable", message)


class NotFoundError(AARError):
    def __init__(self, message: str = "Run not found.") -> None:
        super().__init__("not_found", message)


def safe_error_code(exc: BaseException) -> str:
    if isinstance(exc, AARError):
        return exc.code
    return "internal_error"


def safe_error_message(exc: BaseException) -> str:
    if isinstance(exc, AARError):
        return exc.message
    return "The run failed due to an internal error."
