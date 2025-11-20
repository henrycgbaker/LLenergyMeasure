#!/usr/bin/env python3
"""
Migration script to convert v1.0 configurations to v2.0 format.

Usage:
    python scripts/migrate_config.py <input_file> <output_file>
    python scripts/migrate_config.py configs/a_default_config.py configs_v2/base_config.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_efficiency.config import ExperimentConfig


def load_legacy_config(file_path: Path) -> dict:
    """
    Load legacy configuration from Python file.

    Args:
        file_path: Path to v1.0 config file

    Returns:
        Configuration dictionary
    """
    # For Python config files, execute them
    if file_path.suffix == ".py":
        import importlib.util

        spec = importlib.util.spec_from_file_location("legacy_config", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Try to get base_config
        if hasattr(module, "base_config"):
            return module.base_config
        else:
            raise ValueError(f"No 'base_config' found in {file_path}")

    # For JSON files
    elif file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def migrate_config(legacy_config: dict) -> ExperimentConfig:
    """
    Migrate v1.0 config dictionary to v2.0 ExperimentConfig.

    Args:
        legacy_config: v1.0 configuration dictionary

    Returns:
        v2.0 ExperimentConfig instance
    """
    print("Migrating configuration...")
    print(f"  Model: {legacy_config.get('model_name', 'unknown')}")
    print(f"  Precision: {legacy_config.get('fp_precision', 'unknown')}")

    try:
        config = ExperimentConfig.from_legacy_dict(legacy_config)
        print("✓ Migration successful!")
        return config
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise


def save_config(config: ExperimentConfig, output_path: Path) -> None:
    """
    Save v2.0 configuration to file.

    Args:
        config: ExperimentConfig instance
        output_path: Path to save config
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(config.model_dump(exclude_none=True), f, indent=2)
        print(f"✓ Saved JSON config to: {output_path}")

    # Save as Python file
    elif output_path.suffix == ".py":
        with open(output_path, "w") as f:
            f.write("# Auto-generated v2.0 configuration\n")
            f.write("# Migrated from v1.0\n\n")
            f.write("from llm_efficiency.config import ExperimentConfig\n\n")
            f.write("config = ExperimentConfig(\n")
            for key, value in config.model_dump(exclude_none=True).items():
                f.write(f"    {key}={repr(value)},\n")
            f.write(")\n")
        print(f"✓ Saved Python config to: {output_path}")

    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate v1.0 configurations to v2.0 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate single config
  python scripts/migrate_config.py configs/a_default_config.py configs_v2/base_config.json

  # Migrate to Python format
  python scripts/migrate_config.py old_config.json new_config.py

  # Test migration without saving
  python scripts/migrate_config.py configs/a_default_config.py --dry-run
        """,
    )

    parser.add_argument("input_file", type=Path, help="Input v1.0 configuration file")
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output v2.0 configuration file (JSON or Python)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test migration without saving output",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate migrated config",
    )

    args = parser.parse_args()

    # Check input file exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    try:
        # Load legacy config
        print(f"Loading v1.0 config from: {args.input_file}")
        legacy_config = load_legacy_config(args.input_file)
        print(f"✓ Loaded {len(legacy_config)} configuration keys")

        # Migrate
        new_config = migrate_config(legacy_config)

        # Validate
        if args.validate:
            print("\nValidating migrated configuration...")
            # Validation happens automatically in Pydantic
            print("✓ Configuration is valid!")

        # Display summary
        print("\nMigration Summary:")
        print(f"  Model: {new_config.model_name}")
        print(f"  Precision: {new_config.precision}")
        print(f"  Batch size: {new_config.batching.batch_size}")
        print(f"  Quantization: {new_config.quantization.enabled}")
        print(f"  Decoder mode: {new_config.decoder.mode}")

        # Save if not dry-run
        if not args.dry_run:
            if args.output_file is None:
                print("\nError: output_file required (or use --dry-run)")
                return 1

            save_config(new_config, args.output_file)
            print(f"\n✓ Migration complete!")
        else:
            print("\n✓ Dry-run successful (no files saved)")

        return 0

    except Exception as e:
        print(f"\n✗ Migration failed: {e}", file=sys.stderr)
        if args.validate:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
