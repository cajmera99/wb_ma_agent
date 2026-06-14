from fastapi import Request
from backend.services.app_state import AppState
from backend.services.run_store import RunStore


def get_app_state(request: Request) -> AppState:
    """Inject the shared AppState (loaded at startup) into a route handler."""
    return request.app.state.app_state


def get_run_store(request: Request) -> RunStore:
    """Inject the shared RunStore into a route handler."""
    return request.app.state.run_store
