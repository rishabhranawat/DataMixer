from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

from core.dataset import Dataset

class DataMixingAlgorithm(ABC):
    """Abstract base class for data mixing algorithms."""

    def __init__(self, 
                 budget: int,
                 seed: Optional[int] = None):
        self.budget = budget
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def compute_proportions(self, datasets: List['Dataset'], **kwargs) -> Dict[str, float]:
        """Compute the proportions of each dataset in the mixed dataset.
        
        Args:
            datasets: List of datasets to mix
            **kwargs: Additional arguments specific to the mixing algorithm
        """
        pass
    
    def validate_proportions(self, proportions: Dict[str, float]) -> None:
        return abs(sum(proportions.values()) - 1.0) < 1e-6
