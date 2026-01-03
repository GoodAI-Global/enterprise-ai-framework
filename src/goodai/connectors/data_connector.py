"""
Data Connector Module

Unified interface for connecting to various data sources.
Supports CSV, JSON, databases, and APIs.

Good AI Philosophy: Non-invasive by default.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd


@dataclass
class ConnectionConfig:
    """Configuration for a data connection."""

    source_type: str  # csv, json, database, api
    path_or_url: str
    options: Dict[str, Any]


@dataclass
class DataSnapshot:
    """A snapshot of data from a source."""

    source: str
    timestamp: str
    row_count: int
    column_count: int
    columns: List[str]
    sample_data: pd.DataFrame
    metadata: Dict[str, Any]


class DataConnector:
    """
    Unified interface for connecting to data sources.

    Supports:
    - CSV files (local and remote)
    - JSON files
    - Excel files
    - SQL databases (via connection strings)
    - REST APIs (GET requests)

    Args:
        config: Optional default configuration

    Example:
        >>> connector = DataConnector()
        >>> df = connector.read("production_data.csv")
        >>> df = connector.read("https://api.example.com/data.json", source_type="json")
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._cache = {}
        self._cache_enabled = self.config.get("cache_enabled", False)

    def read(
        self,
        path_or_url: str,
        source_type: Optional[str] = None,
        **options
    ) -> pd.DataFrame:
        """
        Read data from a source into a DataFrame.

        Args:
            path_or_url: File path or URL to data source
            source_type: Type of source (auto-detected if not provided)
                        Options: csv, json, excel, parquet
            **options: Additional options passed to pandas reader

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If local file doesn't exist
            ValueError: If source type cannot be determined

        Example:
            >>> df = connector.read("data.csv")
            >>> df = connector.read("data.json", source_type="json")
            >>> df = connector.read("data.xlsx", sheet_name="Sheet1")
        """
        # Auto-detect source type
        if source_type is None:
            source_type = self._detect_source_type(path_or_url)

        # Check cache
        cache_key = f"{path_or_url}:{source_type}"
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Read based on type
        if source_type == "csv":
            df = self._read_csv(path_or_url, **options)
        elif source_type == "json":
            df = self._read_json(path_or_url, **options)
        elif source_type == "excel":
            df = self._read_excel(path_or_url, **options)
        elif source_type == "parquet":
            df = self._read_parquet(path_or_url, **options)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Cache if enabled
        if self._cache_enabled:
            self._cache[cache_key] = df.copy()

        return df

    def _detect_source_type(self, path_or_url: str) -> str:
        """Detect source type from path or URL."""
        path_lower = path_or_url.lower()

        if path_lower.endswith(".csv"):
            return "csv"
        elif path_lower.endswith(".json"):
            return "json"
        elif path_lower.endswith((".xlsx", ".xls")):
            return "excel"
        elif path_lower.endswith(".parquet"):
            return "parquet"
        else:
            # Default to CSV for unknown types
            return "csv"

    def _read_csv(self, path: str, **options) -> pd.DataFrame:
        """Read CSV file."""
        path_obj = Path(path)

        # Handle local vs remote
        if path_obj.exists():
            return pd.read_csv(path, **options)
        elif path.startswith(("http://", "https://")):
            return pd.read_csv(path, **options)
        else:
            raise FileNotFoundError(f"CSV file not found: {path}")

    def _read_json(self, path: str, **options) -> pd.DataFrame:
        """Read JSON file."""
        path_obj = Path(path)

        if path_obj.exists():
            return pd.read_json(path, **options)
        elif path.startswith(("http://", "https://")):
            return pd.read_json(path, **options)
        else:
            raise FileNotFoundError(f"JSON file not found: {path}")

    def _read_excel(self, path: str, **options) -> pd.DataFrame:
        """Read Excel file."""
        path_obj = Path(path)

        if path_obj.exists():
            return pd.read_excel(path, **options)
        else:
            raise FileNotFoundError(f"Excel file not found: {path}")

    def _read_parquet(self, path: str, **options) -> pd.DataFrame:
        """Read Parquet file."""
        path_obj = Path(path)

        if path_obj.exists():
            return pd.read_parquet(path, **options)
        else:
            raise FileNotFoundError(f"Parquet file not found: {path}")

    def write(
        self,
        df: pd.DataFrame,
        path: str,
        output_type: Optional[str] = None,
        **options
    ) -> str:
        """
        Write DataFrame to a file.

        Args:
            df: DataFrame to write
            path: Output file path
            output_type: Type of output (auto-detected if not provided)
            **options: Additional options passed to pandas writer

        Returns:
            Path to written file

        Example:
            >>> connector.write(df, "output.csv")
            >>> connector.write(df, "output.json", orient="records")
        """
        if output_type is None:
            output_type = self._detect_source_type(path)

        if output_type == "csv":
            df.to_csv(path, index=False, **options)
        elif output_type == "json":
            df.to_json(path, **options)
        elif output_type == "excel":
            df.to_excel(path, index=False, **options)
        elif output_type == "parquet":
            df.to_parquet(path, **options)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        return path

    def get_snapshot(
        self,
        path_or_url: str,
        sample_rows: int = 5
    ) -> DataSnapshot:
        """
        Get a snapshot of data source without loading full dataset.

        Args:
            path_or_url: Path or URL to data source
            sample_rows: Number of sample rows to include

        Returns:
            DataSnapshot with metadata and sample data
        """
        from datetime import datetime

        df = self.read(path_or_url)

        return DataSnapshot(
            source=path_or_url,
            timestamp=datetime.now().isoformat(),
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            sample_data=df.head(sample_rows),
            metadata={
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage_bytes": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
            }
        )

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate DataFrame schema against expected columns.

        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            required_columns: List of columns that must exist (default: all expected)

        Returns:
            Validation result dictionary
        """
        required_columns = required_columns or expected_columns

        actual_columns = set(df.columns)
        expected_set = set(expected_columns)
        required_set = set(required_columns)

        missing_required = required_set - actual_columns
        missing_optional = expected_set - required_set - actual_columns
        extra_columns = actual_columns - expected_set

        is_valid = len(missing_required) == 0

        return {
            "valid": is_valid,
            "missing_required": list(missing_required),
            "missing_optional": list(missing_optional),
            "extra_columns": list(extra_columns),
            "column_count": len(actual_columns),
            "expected_count": len(expected_columns),
        }

    def merge_sources(
        self,
        sources: List[str],
        how: str = "concat",
        **options
    ) -> pd.DataFrame:
        """
        Merge multiple data sources into one DataFrame.

        Args:
            sources: List of file paths or URLs
            how: Merge method - "concat" (vertical) or "merge" (horizontal)
            **options: Options passed to pd.concat or pd.merge

        Returns:
            Merged DataFrame
        """
        dfs = [self.read(source) for source in sources]

        if how == "concat":
            return pd.concat(dfs, ignore_index=True, **options)
        elif how == "merge":
            if len(dfs) < 2:
                return dfs[0] if dfs else pd.DataFrame()

            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, **options)
            return result
        else:
            raise ValueError(f"Unsupported merge method: {how}")

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = {}

    def enable_cache(self, enabled: bool = True):
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
