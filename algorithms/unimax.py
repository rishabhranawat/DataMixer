from core.base import DataMixingAlgorithm
from core.dataset import Dataset
from typing import Dict, List, Optional, Any

class UniMax(DataMixingAlgorithm):
    """UniMax is a data mixing algorithm that mixes datasets according to the UniMax algorithm."""

    def __init__(self, 
                 budget: int, 
                 seed: Optional[int] = None,
                 max_epochs_per_language: Optional[int] = None):
        super().__init__(budget, seed)
        self.max_epochs_per_language = max_epochs_per_language

    def compute_proportions(self, datasets: List[Dataset]) -> Dict[str, float]:
        """Compute the proportions of each dataset in the mixed dataset."""
        proportions = {}
        L = len(datasets)
        # Sort datasets by size in descending order
        sorted_datasets = sorted(datasets, key=lambda x: x.size, reverse=True)
        
        remaining_budget = self.budget
        for i, dataset in enumerate(sorted_datasets):
            uniform_budget = remaining_budget / (L - i)
            if uniform_budget > dataset.size * self.max_epochs_per_language:
                proportions[dataset.name] = dataset.size * self.max_epochs_per_language
            else:
                proportions[dataset.name] = uniform_budget
            
            remaining_budget -= proportions[dataset.name]

        # Normalize proportions
        total_proportion = sum(proportions.values())
        proportions = {k: v / total_proportion for k, v in proportions.items()}
        return proportions