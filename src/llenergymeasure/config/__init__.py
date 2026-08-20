"""Configuration subsystem for llenergymeasure.

Public API:
- ExperimentConfig: Main experiment configuration model
- load_experiment_config: Load and validate a single-experiment YAML/JSON file
- load_user_config: Load user preferences from XDG config dir
- get_user_config_path: Return the XDG user config path
- SchemaLoader / DiscoveredSchema: Access discovered engine parameter schemas
"""

from llenergymeasure.config.loader import (
    deep_merge,
    load_experiment_config,
)
from llenergymeasure.config.models import (
    BaselineConfig,
    DatasetConfig,
    ExperimentConfig,
    MeasurementConfig,
    OfflineSection,
    ServerWarmupConfig,
    TaskConfig,
    WarmupConfig,
)
from llenergymeasure.config.schema_loader import (
    DiscoveredSchema,
    DiscoveryLimitation,
    SchemaLoader,
    UnsupportedSchemaVersionError,
)
from llenergymeasure.config.user_config import (
    UserConfig,
    get_user_config_path,
    load_user_config,
)

__all__ = [
    "BaselineConfig",
    "DatasetConfig",
    "DiscoveredSchema",
    "DiscoveryLimitation",
    "ExperimentConfig",
    "MeasurementConfig",
    "OfflineSection",
    "SchemaLoader",
    "ServerWarmupConfig",
    "TaskConfig",
    "UnsupportedSchemaVersionError",
    "UserConfig",
    "WarmupConfig",
    "deep_merge",
    "get_user_config_path",
    "load_experiment_config",
    "load_user_config",
]
