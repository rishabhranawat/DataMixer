from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

@dataclass
class Dataset:
    """Represents a dataset in the mixing process."""
    name: str
    size: int
    samples: List[Any]