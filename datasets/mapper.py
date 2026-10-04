"""
datasets/mapper.py

Maps dataset files to benchmark tasks. Provides utilities to load datasets 
and match them with the appropriate task classes.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass

# Dynamically import task classes
from benchmarks.tasks.bug_fixing import BugFixingBenchmark
from benchmarks.tasks.code_generation import CodeGenerationBenchmark
from benchmarks.tasks.code_review import CodeReviewBenchmark
from benchmarks.tasks.refactoring import RefactoringBenchmark
from benchmarks.tasks.test_generation import TestGenerationBenchmark
from benchmarks.tasks.translation import TranslationBenchmark


@dataclass
class DatasetRecord:
    """Represents a single record from a dataset CSV."""
    task_name: str
    language: str
    record_id: str
    data: Dict[str, Any]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {"task": self.task_name, "language": self.language, "id": self.record_id, "data": self.data},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DatasetMapper:
    """Maps datasets to benchmark tasks and provides utilities for loading data."""

    # Task to dataset directory mapping
    TASK_DATASET_MAP = {
        "bug_fixing": {
            "dir": "bug_fixing",
            "class": BugFixingBenchmark,
            "languages": ["python", "cpp", "javascript"],
            "csv_columns": ["id", "buggy_code", "fixed_code", "expected_output"],
        },
        "code_generation": {
            "dir": "code_generation",
            "class": CodeGenerationBenchmark,
            "languages": ["python", "cpp", "javascript"],
            "csv_columns": {"python": ["id", "prompt", "completed", "test", "entry_point"], "cpp": ["id", "instruction", "generated_code", "expected_output"], "javascript": ["id", "instruction", "generated_code", "expected_output"]},
        },
        "code_review": {
            "dir": "code_review",
            "class": CodeReviewBenchmark,
            "languages": ["python", "cpp", "javascript"],
            "csv_columns": ["id", "code_snippet", "review", "expected_reviews"],
        },
        "refactoring": {
            "dir": "refactoring",
            "class": RefactoringBenchmark,
            "languages": ["python", "cpp", "javascript"],
            "csv_columns": ["id", "original_code", "refactoring_task", "output_expected", "expected_console_output", "test_harness"],
        },
        "test_generation": {
            "dir": "test_generation",
            "class": TestGenerationBenchmark,
            "languages": ["python"],
            "csv_columns": ["id", "code_snippet", "description", "test", "entry_point"],
        },
        "translation": {
            "dir": "translation",
            "class": TranslationBenchmark,
            "languages": ["python", "cpp", "javascript"],
            "csv_columns": ["id", "code_snippet", "expected_output"],
        },
    }

    def __init__(self, dataset_root: Optional[Path] = None):
        """
        Initialize the mapper with a dataset root directory.
        
        Args:
            dataset_root: Path to the datasets directory. If None, uses current module's parent.
        """
        if dataset_root is None:
            dataset_root = Path(__file__).parent
        self.dataset_root = dataset_root

    def get_dataset_path(self, task_name: str, language: str) -> Path:
        """
        Get the path to a specific dataset file.
        
        Args:
            task_name: Name of the task (e.g., "bug_fixing")
            language: Programming language (e.g., "python")
            
        Returns:
            Path to the dataset CSV file
            
        Raises:
            ValueError: If task or language is not supported
        """
        if task_name not in self.TASK_DATASET_MAP:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(self.TASK_DATASET_MAP.keys())}"
            )
        
        task_config = self.TASK_DATASET_MAP[task_name]
        if language not in task_config["languages"]:
            raise ValueError(
                f"Language '{language}' not supported for task '{task_name}'. "
                f"Available: {task_config['languages']}"
            )
        
        # Map language to CSV filename pattern
        filename = f"{language}.csv"
        
        dataset_path = self.dataset_root / task_config["dir"] / filename
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        return dataset_path

    def get_task_class(self, task_name: str) -> Type:
        """
        Get the benchmark task class for a given task name.
        
        Args:
            task_name: Name of the task (e.g., "bug_fixing")
            
        Returns:
            The benchmark class
            
        Raises:
            ValueError: If task is not found
        """
        if task_name not in self.TASK_DATASET_MAP:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(self.TASK_DATASET_MAP.keys())}"
            )
        
        return self.TASK_DATASET_MAP[task_name]["class"]

    def load_dataset(
        self, 
        task_name: str, 
        language: str,
        limit: Optional[int] = None
    ) -> List[DatasetRecord]:
        """
        Load all records from a dataset.
        
        Args:
            task_name: Name of the task
            language: Programming language
            limit: Maximum number of records to load (None for all)
            
        Returns:
            List of DatasetRecord objects
        """
        dataset_path = self.get_dataset_path(task_name, language)
        records = []
        contracts_path = dataset_path.with_name(f"{dataset_path.stem}_contracts.json")
        contracts = json.loads(contracts_path.read_text(encoding="utf-8")) if contracts_path.exists() else {}
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                
                row_data = dict(row)
                if str(row.get("id", str(i))) in contracts:
                    row_data["_facets_contract"] = contracts[str(row.get("id", str(i)))]
                record = DatasetRecord(
                    task_name=task_name,
                    language=language,
                    record_id=row.get("id", str(i)),
                    data=row_data
                )
                records.append(record)
        
        return records

    def load_record(
        self, 
        task_name: str, 
        language: str, 
        record_id: str
    ) -> Optional[DatasetRecord]:
        """
        Load a specific record from a dataset.
        
        Args:
            task_name: Name of the task
            language: Programming language
            record_id: ID of the record to load
            
        Returns:
            DatasetRecord if found, None otherwise
        """
        dataset_path = self.get_dataset_path(task_name, language)
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == record_id:
                    return DatasetRecord(
                        task_name=task_name,
                        language=language,
                        record_id=record_id,
                        data=dict(row)
                    )
        
        return None

    def get_available_tasks(self) -> List[str]:
        """Get list of all available task names."""
        return list(self.TASK_DATASET_MAP.keys())

    def get_available_languages(self, task_name: str) -> List[str]:
        """
        Get list of available languages for a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            List of language codes
            
        Raises:
            ValueError: If task is not found
        """
        if task_name not in self.TASK_DATASET_MAP:
            raise ValueError(f"Unknown task '{task_name}'")
        
        return self.TASK_DATASET_MAP[task_name]["languages"]

    def get_dataset_info(self, task_name: str, language: str) -> Dict[str, Any]:
        """
        Get metadata information about a dataset.
        
        Args:
            task_name: Name of the task
            language: Programming language
            
        Returns:
            Dictionary with dataset information including record count
        """
        dataset_path = self.get_dataset_path(task_name, language)
        task_config = self.TASK_DATASET_MAP[task_name]
        
        records = self.load_dataset(task_name, language)
        ids = [record.record_id for record in records]
        contracts_path = dataset_path.with_name(f"{dataset_path.stem}_contracts.json")
        hash_material = dataset_path.read_bytes() + (contracts_path.read_bytes() if contracts_path.exists() else b"")
        file_hash = hashlib.sha256(hash_material).hexdigest()
        
        return {
            "task_name": task_name,
            "language": language,
            "csv_path": str(dataset_path),
            "record_count": len(records),
            "record_ids_unique": len(ids) == len(set(ids)),
            "dataset_hash": file_hash,
            "columns": task_config["csv_columns"],
            "benchmark_class": task_config["class"].__name__,
        }

    def map_record_to_benchmark_kwargs(
        self, 
        record: DatasetRecord
    ) -> Dict[str, Any]:
        """
        Convert a dataset record to kwargs suitable for benchmark execution.
        
        Args:
            record: A DatasetRecord from a dataset
            
        Returns:
            Dictionary with kwargs for the benchmark task
        """
        task_name = record.task_name
        data = record.data
        # language comes form SuiteConfig
     
        # Base kwargs that apply to all tasks
        kwargs: Dict[str, Any] = {}
        kwargs["dataset_hash"] = record.fingerprint
        
        # Task-specific mapping
        if task_name == "bug_fixing":
            kwargs.update({
                "code_input": data.get("buggy_code"),
                "expected_output": data.get("expected_output"),
            })
        
        elif task_name == "code_generation":
            kwargs.update({
                "code_input": data.get("prompt") or data.get("instruction"),
                "reference_implementation": data.get("completed") or data.get("expected_output"),
                "test": data.get("test", ""),
                "entry_point": data.get("entry_point", ""),
            })
        
        elif task_name == "code_review":
            kwargs.update({
                "code_input": data.get("code_snippet"),
                "expected_output": data.get("review"),
                "expected_reviews": data.get("expected_reviews", "").split("|") if data.get("expected_reviews") else [],
            })
        
        elif task_name == "refactoring":
            contract = data.get("_facets_contract") or {}
            kwargs.update({
                "code_input": data.get("original_code"),
                "refactoring_task": data.get("refactoring_task"),
                "expected_output": data.get("output_expected"),  # compatibility
                "reference_implementation": data.get("output_expected"),
                "expected_output_is_implementation": True,
                "expected_console_output": contract.get("expected_console_output", data.get("expected_console_output")),
                "test_harness": contract.get("test_harness", data.get("test_harness", "")),
                "test_contract": contract.get("contract", "exact_console_output"),
            })

        elif task_name == "test_generation":
            kwargs.update({
                "code_input": data.get("code_snippet"),
                "description": data.get("description"),
                "reference_tests": data.get("test", ""),
                "entry_point": data.get("entry_point", ""),
            })
        
        elif task_name == "translation":
            kwargs.update({
                "code_input": data.get("code_snippet"),
                "expected_output": data.get("expected_output"),
            })
        
        return kwargs


# Global mapper instance
_mapper_instance: Optional[DatasetMapper] = None


def get_mapper(dataset_root: Optional[Path] = None) -> DatasetMapper:
    """
    Get or create a global DatasetMapper instance.
    
    Args:
        dataset_root: Path to datasets directory (only used on first call)
        
    Returns:
        DatasetMapper instance
    """
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = DatasetMapper(dataset_root)
    return _mapper_instance


# Convenience functions
def load_dataset(
    task_name: str,
    language: str, 
    limit: Optional[int] = None
) -> List[DatasetRecord]:
    """Load a dataset using the global mapper."""
    return get_mapper().load_dataset(task_name, language, limit)


def get_dataset_path(task_name: str, language: str) -> Path:
    """Get dataset path using the global mapper."""
    return get_mapper().get_dataset_path(task_name, language)


def get_task_class(task_name: str) -> Type:
    """Get task class using the global mapper."""
    return get_mapper().get_task_class(task_name)
