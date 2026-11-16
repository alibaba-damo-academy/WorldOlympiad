"""Domain-agnostic task management and workflow coordination tools.

This module provides TODO list management capabilities for systematic task
tracking. Supports adding, completing, querying, and listing tasks with
timestamps and status tracking for organized workflow execution.
"""

import logging
import time

from dataclasses import dataclass
from typing import Any

from agents import function_tool

from scenecode.agent_utils.response_datatypes import JSONSerializable

console_logger = logging.getLogger(__name__)


@dataclass
class TodoItem(JSONSerializable):
    """Represents a todo item."""

    id: str
    """Unique identifier for the todo item."""
    task: str
    """Description of the task to be completed."""
    status: str
    """Current status ('pending' or 'completed')."""
    created_at: float
    """Timestamp when the item was created."""
    completed_at: float | None = None
    """Timestamp when the item was completed (if applicable)."""


@dataclass
class TodoSummary(JSONSerializable):
    """Summary of todo list statistics."""

    total: int
    """Total number of todo items."""
    pending: int
    """Number of pending todo items."""
    completed: int
    """Number of completed todo items."""


@dataclass
class TodoOperationResult(JSONSerializable):
    """Result of TODO list operations."""

    success: bool
    """Whether the operation succeeded."""
    action: str
    """The action that was performed."""
    message: str | None = None
    """Status or error message."""
    task: TodoItem | None = None
    """Single task result (for add/complete/get_next actions)."""
    todos: list[TodoItem] | None = None
    """All todos (for list_all action)."""
    summary: TodoSummary | None = None
    """Summary statistics (for list_all action)."""


class WorkflowTools:
    """Tools for managing designer workflow and task tracking."""

    def __init__(self) -> None:
        """Initialize workflow tools."""
        self._designer_todos: list[TodoItem] = []
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create closure-based tools that capture self."""

        @function_tool
        def designer_todo_manager(action: str, task: str | None = None) -> str:
            """Track your design tasks.

            Use this to keep track of what you need to do during the design process.
            Add tasks as you think of them, mark them complete when done, or check
            what's next to work on.

            Args:
                action: What to do - 'add' for new tasks, 'complete' to mark the
                       latest task done, 'get_next' to see what to do next, or
                       'list_all' to see everything.
                task: Description of the task (only needed when adding).

            Returns:
                Confirmation of the action and current task status.
            """
            return self._designer_todo_manager_impl(action=action, task=task)

        return {
            "designer_todo_manager": designer_todo_manager,
        }

    def _designer_todo_manager_impl(self, action: str, task: str | None = None) -> str:
        """Manage designer's internal TODO list.

        Args:
            action: Action to perform ('add', 'complete', 'get_next', 'list_all').
            task: Task description (required for 'add' action).

        Returns:
            JSON string with TodoOperationResult.
        """
        console_logger.info(
            f"Tool called: designer_todo_manager(action={action!r}, task={task!r})"
        )
        if action == "add":
            if not task:
                return TodoOperationResult(
                    success=False,
                    action=action,
                    message="Task required for 'add' action",
                ).to_json()

            todo_item = TodoItem(
                id=f"todo_{len(self._designer_todos)}",
                task=task,
                status="pending",
                created_at=time.time(),
            )
            self._designer_todos.append(todo_item)
            return TodoOperationResult(
                success=True,
                action=action,
                message=f"Added TODO: {task}",
                task=todo_item,
            ).to_json()

        elif action == "complete":
            for todo in reversed(self._designer_todos):
                if todo.status == "pending":
                    todo.status = "completed"
                    todo.completed_at = time.time()
                    return TodoOperationResult(
                        success=True,
                        action=action,
                        message=f"Completed: {todo.task}",
                        task=todo,
                    ).to_json()
            return TodoOperationResult(
                success=False, action=action, message="No pending tasks to complete"
            ).to_json()

        elif action == "get_next":
            for todo in self._designer_todos:
                if todo.status == "pending":
                    return TodoOperationResult(
                        success=True, action=action, task=todo
                    ).to_json()
            return TodoOperationResult(
                success=False, action=action, message="No pending tasks"
            ).to_json()

        elif action == "list_all":
            summary = TodoSummary(
                total=len(self._designer_todos),
                pending=len([t for t in self._designer_todos if t.status == "pending"]),
                completed=len(
                    [t for t in self._designer_todos if t.status == "completed"]
                ),
            )
            return TodoOperationResult(
                success=True, action=action, todos=self._designer_todos, summary=summary
            ).to_json()

        return TodoOperationResult(
            success=False, action=action, message=f"Unknown action: {action}"
        ).to_json()
