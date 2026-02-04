"""Container lifecycle management for persistent container strategy.

Manages the lifecycle of long-running Docker containers using docker compose up/exec/down.
This is an alternative to the default ephemeral strategy (docker compose run --rm).

Persistent mode trades isolation for speed:
- Containers stay warm between experiments
- No container startup overhead per experiment
- Requires explicit teardown
- GPU memory may need manual clearing between experiments
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

# Type alias for status callback: (service_name, status) -> None
StatusCallback = Callable[[str, str], None]


@dataclass
class ContainerState:
    """State of a managed container."""

    service: str
    status: Literal["stopped", "starting", "running", "error"]
    restart_count: int = 0


@dataclass
class ContainerManager:
    """Manages persistent Docker container lifecycle.

    Usage:
        manager = ContainerManager(services=["pytorch", "vllm"])
        manager.start_all()  # docker compose up -d pytorch vllm

        # For each experiment:
        result = manager.exec("vllm", ["lem", "experiment", "config.yaml"])

        manager.stop_all()  # docker compose down
    """

    services: list[str]
    warmup_delay: float = 0.0
    auto_teardown: bool = True
    _states: dict[str, ContainerState] = field(default_factory=dict, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize container states."""
        for service in self.services:
            self._states[service] = ContainerState(service=service, status="stopped")

    def start_all(self, status_callback: StatusCallback | None = None) -> bool:
        """Start all managed containers.

        Args:
            status_callback: Optional callback(service, status) for progress updates.
                Called with status "starting" before and "ready"/"failed" after each service.

        Returns:
            True if all containers started successfully.
        """
        if self._started:
            logger.debug("Containers already started")
            return True

        logger.info("Starting containers: {}", ", ".join(self.services))

        # Notify starting for all services
        for service in self.services:
            self._states[service].status = "starting"
            if status_callback:
                status_callback(service, "starting")

        cmd = ["docker", "compose", "up", "-d", *self.services]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("Failed to start containers: {}", result.stderr)
            for service in self.services:
                self._states[service].status = "error"
                if status_callback:
                    status_callback(service, "failed")
            return False

        for service in self.services:
            self._states[service].status = "running"
            if status_callback:
                status_callback(service, "ready")

        self._started = True

        # Optional warmup delay after container start
        if self.warmup_delay > 0:
            import time

            logger.debug("Waiting {}s for container warmup", self.warmup_delay)
            time.sleep(self.warmup_delay)

        return True

    def exec(
        self,
        service: str,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute command in a running container.

        Args:
            service: Service name to exec into.
            command: Command and arguments to execute.
            env: Environment variables to pass (-e flags).

        Returns:
            CompletedProcess with return code and output.
        """
        if service not in self._states:
            msg = f"Unknown service: {service}"
            raise ValueError(msg)

        if self._states[service].status != "running":
            logger.warning(
                "Container {} not running (status: {}), attempting to start",
                service,
                self._states[service].status,
            )
            if not self.start_all():
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=1,
                    stdout="",
                    stderr="Container start failed",
                )

        # Build exec command with optional env vars
        cmd = ["docker", "compose", "exec"]

        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([service, *command])

        logger.debug("Executing in {}: {}", service, " ".join(command[:3]))
        return subprocess.run(cmd, capture_output=False, text=True, check=False)

    def stop_all(self, status_callback: StatusCallback | None = None) -> bool:
        """Stop all managed containers.

        Args:
            status_callback: Optional callback(service, status) for progress updates.
                Called with status "stopping" before and "stopped"/"failed" after.

        Returns:
            True if all containers stopped successfully.
        """
        if not self._started:
            logger.debug("Containers not started, nothing to stop")
            return True

        logger.info("Stopping containers: {}", ", ".join(self.services))

        # Notify stopping for all services
        for service in self.services:
            if status_callback:
                status_callback(service, "stopping")

        cmd = ["docker", "compose", "down"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("Failed to stop containers: {}", result.stderr)
            for service in self.services:
                if status_callback:
                    status_callback(service, "failed")
            return False

        for service in self.services:
            self._states[service].status = "stopped"
            if status_callback:
                status_callback(service, "stopped")

        self._started = False
        return True

    def get_status(self, service: str) -> str:
        """Get current status of a service.

        Args:
            service: Service name to query.

        Returns:
            Status string: 'stopped', 'starting', 'running', or 'error'.

        Raises:
            ValueError: If service is not managed by this manager.
        """
        if service not in self._states:
            msg = f"Unknown service: {service}"
            raise ValueError(msg)
        return self._states[service].status

    def restart_service(self, service: str) -> bool:
        """Restart a single service.

        Args:
            service: Service name to restart.

        Returns:
            True if restart successful.
        """
        if service not in self._states:
            msg = f"Unknown service: {service}"
            raise ValueError(msg)

        state = self._states[service]
        state.restart_count += 1

        logger.info("Restarting container {} (restart #{})", service, state.restart_count)

        cmd = ["docker", "compose", "restart", service]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("Failed to restart {}: {}", service, result.stderr)
            state.status = "error"
            return False

        state.status = "running"
        return True

    def __enter__(self) -> ContainerManager:
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit - auto teardown if enabled."""
        if self.auto_teardown:
            self.stop_all()


__all__ = ["ContainerManager", "ContainerState"]
