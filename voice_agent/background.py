"""
Lightweight async background task manager.

Used for fire-and-forget operations that should run alongside the
live session (e.g. cache pre-fetching, periodic refreshes) without
blocking tool responses or audio streaming.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages a set of tracked ``asyncio.Task`` objects.

    • Tasks that raise exceptions are logged but never crash the session.
    • Completed tasks are automatically pruned from the tracking set.
    • ``cancel_all()`` should be called on session teardown.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    # ── Public API ──────────────────────────────────────────────

    def schedule(self, coro: Awaitable[Any], *, name: str | None = None) -> asyncio.Task[Any]:
        """Fire-and-forget *coro*; returns the created task."""
        task = asyncio.create_task(self._safe_wrapper(coro), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.debug("Scheduled background task: %s", task.get_name())
        return task

    def schedule_recurring(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        interval_s: float,
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        """Run *coro_factory()* every *interval_s* seconds until cancelled."""

        async def _loop() -> None:
            while True:
                await self._safe_wrapper(coro_factory())
                await asyncio.sleep(interval_s)

        task = asyncio.create_task(_loop(), name=name or "recurring")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.debug(
            "Scheduled recurring task every %.1fs: %s",
            interval_s,
            task.get_name(),
        )
        return task

    async def cancel_all(self) -> None:
        """Cancel every tracked task and wait for them to finish."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.debug("All background tasks cancelled")

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    async def _safe_wrapper(coro: Awaitable[Any]) -> Any:
        """Await *coro*, logging (but swallowing) any exception."""
        try:
            return await coro
        except asyncio.CancelledError:
            raise  # let cancellation propagate
        except Exception:
            logger.exception("Background task failed")
            return None
