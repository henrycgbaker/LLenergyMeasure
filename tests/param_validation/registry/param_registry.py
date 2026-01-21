"""Central registry for all parameter specifications.

Provides a singleton registry for registering, querying, and managing
ParamSpec definitions across all backends.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from .models import HardwareRequirement, ParamSpec, VerificationType


class ParamRegistry:
    """Central registry for ParamSpec definitions.

    Singleton that stores all registered parameter specifications and
    provides query/filter methods for test generation.
    """

    _instance: ParamRegistry | None = None

    def __new__(cls) -> ParamRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._specs: dict[str, ParamSpec] = {}
            cls._instance._by_backend: dict[str, list[str]] = defaultdict(list)
            cls._instance._by_category: dict[str, list[str]] = defaultdict(list)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls._instance = None

    def register(self, spec: ParamSpec, *, allow_duplicate: bool = True) -> bool:
        """Register a ParamSpec.

        Args:
            spec: The ParamSpec to register.
            allow_duplicate: If True, silently skip already-registered specs.
                            If False, raise ValueError on duplicate.

        Returns:
            True if the spec was registered, False if it was already present.

        Raises:
            ValueError: If allow_duplicate is False and spec is already registered.
        """
        if spec.full_name in self._specs:
            if allow_duplicate:
                return False
            raise ValueError(f"ParamSpec '{spec.full_name}' is already registered")

        self._specs[spec.full_name] = spec
        self._by_backend[spec.backend].append(spec.full_name)
        self._by_category[spec.category].append(spec.full_name)
        return True

    def register_all(self, specs: list[ParamSpec]) -> None:
        """Register multiple ParamSpecs."""
        for spec in specs:
            self.register(spec)

    def get(self, full_name: str) -> ParamSpec | None:
        """Get a ParamSpec by its full name."""
        return self._specs.get(full_name)

    def __getitem__(self, full_name: str) -> ParamSpec:
        """Get a ParamSpec by its full name (raises KeyError if not found)."""
        return self._specs[full_name]

    def __contains__(self, full_name: str) -> bool:
        """Check if a ParamSpec is registered."""
        return full_name in self._specs

    def __iter__(self) -> Iterator[ParamSpec]:
        """Iterate over all registered ParamSpecs."""
        return iter(self._specs.values())

    def __len__(self) -> int:
        """Return the number of registered ParamSpecs."""
        return len(self._specs)

    @property
    def all_specs(self) -> list[ParamSpec]:
        """Get all registered ParamSpecs."""
        return list(self._specs.values())

    @property
    def all_names(self) -> list[str]:
        """Get all registered ParamSpec names."""
        return list(self._specs.keys())

    @property
    def backends(self) -> list[str]:
        """Get all backends with registered specs."""
        return list(self._by_backend.keys())

    @property
    def categories(self) -> list[str]:
        """Get all categories with registered specs."""
        return list(self._by_category.keys())

    def by_backend(self, backend: str) -> list[ParamSpec]:
        """Get all ParamSpecs for a specific backend."""
        return [self._specs[name] for name in self._by_backend.get(backend, [])]

    def by_category(self, category: str) -> list[ParamSpec]:
        """Get all ParamSpecs for a specific category."""
        return [self._specs[name] for name in self._by_category.get(category, [])]

    def by_verification_type(self, vtype: VerificationType) -> list[ParamSpec]:
        """Get all ParamSpecs with a specific verification type."""
        return [s for s in self._specs.values() if s.verification_type == vtype]

    def filter(
        self,
        backend: str | None = None,
        category: str | None = None,
        verification_type: VerificationType | None = None,
        hardware_requirements: set[HardwareRequirement] | None = None,
        energy_impact: bool | None = None,
        mockable_only: bool = False,
        exclude_skipped: bool = True,
    ) -> list[ParamSpec]:
        """Filter ParamSpecs by multiple criteria.

        Args:
            backend: Filter by backend name.
            category: Filter by category name.
            verification_type: Filter by verification type.
            hardware_requirements: Only include specs with these requirements as subset.
            energy_impact: Filter by energy impact flag.
            mockable_only: Only include specs that can be tested via mocking.
            exclude_skipped: Exclude specs with skip_reason set.

        Returns:
            List of ParamSpecs matching all criteria.
        """
        specs = list(self._specs.values())

        if backend is not None:
            specs = [s for s in specs if s.backend == backend]

        if category is not None:
            specs = [s for s in specs if s.category == category]

        if verification_type is not None:
            specs = [s for s in specs if s.verification_type == verification_type]

        if hardware_requirements is not None:
            specs = [s for s in specs if s.hardware_requirements <= hardware_requirements]

        if energy_impact is not None:
            specs = [s for s in specs if s.energy_impact == energy_impact]

        if mockable_only:
            specs = [s for s in specs if s.can_mock()]

        if exclude_skipped:
            specs = [s for s in specs if s.skip_reason is None]

        return specs

    def runnable_with_hardware(
        self,
        backend: str | None = None,
    ) -> list[ParamSpec]:
        """Get all ParamSpecs that can run on current hardware.

        Uses hardware_caps to check which requirements are met.

        Args:
            backend: Optionally filter by backend.

        Returns:
            List of runnable ParamSpecs.
        """
        from .hardware_caps import check_requirements

        specs = self.by_backend(backend) if backend else list(self._specs.values())

        runnable = []
        for spec in specs:
            if spec.skip_reason:
                continue
            met, _ = check_requirements(spec.hardware_requirements)
            if met:
                runnable.append(spec)

        return runnable


# Module-level registry instance
registry = ParamRegistry()


def register(spec: ParamSpec) -> ParamSpec:
    """Register a ParamSpec with the global registry.

    Can be used as a decorator-style function:
        spec = register(ParamSpec(...))
    """
    registry.register(spec)
    return spec


def register_all(specs: list[ParamSpec]) -> None:
    """Register multiple ParamSpecs with the global registry."""
    registry.register_all(specs)


def get_registry() -> ParamRegistry:
    """Get the global ParamRegistry instance."""
    return registry
