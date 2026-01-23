"""Tests for protocol definitions."""

from pathlib import Path
from typing import Any

from llenergymeasure.domain.metrics import EnergyMetrics
from llenergymeasure.protocols import (
    EnergyBackend,
    InferenceEngine,
    ModelLoader,
    ResultsRepository,
)


class TestProtocolsAreRuntimeCheckable:
    """Verify protocols can be checked at runtime."""

    def test_model_loader_protocol(self):
        class ConcreteLoader:
            def load(self, config: Any) -> tuple[Any, Any]:
                return (None, None)

        loader = ConcreteLoader()
        assert isinstance(loader, ModelLoader)

    def test_inference_engine_protocol(self):
        class ConcreteEngine:
            def run(
                self,
                model: Any,
                tokenizer: Any,
                prompts: list[str],
                config: Any,
            ) -> Any:
                return {}

        engine = ConcreteEngine()
        assert isinstance(engine, InferenceEngine)

    def test_energy_backend_protocol(self):
        class ConcreteBackend:
            @property
            def name(self) -> str:
                return "test"

            def start_tracking(self) -> Any:
                return None

            def stop_tracking(self, tracker: Any) -> EnergyMetrics:
                return EnergyMetrics(total_energy_j=0, duration_sec=0)

            def is_available(self) -> bool:
                return True

        backend = ConcreteBackend()
        assert isinstance(backend, EnergyBackend)

    def test_results_repository_protocol(self):
        class ConcreteRepository:
            def save_raw(self, experiment_id: str, result: Any) -> Path:
                return Path("test.json")

            def list_raw(self, experiment_id: str) -> list[Path]:
                return []

            def load_raw(self, path: Path) -> Any:
                return None

            def save_aggregated(self, result: Any) -> Path:
                return Path("agg.json")

        repo = ConcreteRepository()
        assert isinstance(repo, ResultsRepository)


class TestNonCompliantImplementations:
    """Verify non-compliant implementations are not recognized."""

    def test_missing_method_not_recognized(self):
        class IncompleteLoader:
            pass  # Missing load method

        loader = IncompleteLoader()
        assert not isinstance(loader, ModelLoader)

    def test_wrong_signature_still_recognized(self):
        # Note: runtime_checkable only checks method existence, not signature
        class WrongSignature:
            def load(self) -> None:  # Wrong signature
                pass

        loader = WrongSignature()
        # This will still pass runtime_checkable (it only checks name exists)
        assert isinstance(loader, ModelLoader)
