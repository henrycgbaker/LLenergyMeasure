"""Warning categories emitted by the config layer."""

from __future__ import annotations


class ConfigValidationWarning(UserWarning):
    """Emitted when config validation finds a recoverable issue.

    Raised, for example, when an engine config section carries an unknown key
    that closely matches a known field (a "did you mean?" suggestion). Distinct
    subclass so callers can elevate only config-validation warnings to errors
    via ``warnings.simplefilter("error", ConfigValidationWarning)``.
    """
