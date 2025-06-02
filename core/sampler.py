from core.base import DataMixingAlgorithm
from core.dataset import Dataset
from typing import Dict, List, Optional, Any
import numpy as np

class DataMixer:

    def __init__(self, 
                 datasets: List[Dataset], 
                 algorithm: DataMixingAlgorithm,
                 seed: Optional[int] = None):
        self._datasets = datasets
        self._algorithm = algorithm
        self._proportions = None
        self._rng = np.random.default_rng(seed)

    @property
    def datasets(self) -> List[Dataset]:
        return self._datasets

    @property
    def algorithm(self) -> DataMixingAlgorithm:
        return self._algorithm

    @property
    def proportions(self) -> Optional[Dict[str, float]]:
        return self._proportions

    @property
    def rng(self) -> np.random.Generator:
        return self._rng


    def mix(self) -> Dataset:
        """Mix the datasets according to the algorithm."""
        proportions = self._algorithm.compute_proportions(self._datasets)
        self._proportions = proportions
    
       
    def sample(self, datasets: List[Dataset], output_size: int) -> Dataset:
        """Sample from multiple datasets according to the given proportions.
        
        Args:
            datasets: List of datasets to sample from
            proportions: Dictionary mapping dataset names to their proportions
            output_size: Total number of samples in the output dataset
            
        Returns:
            A new Dataset containing the mixed samples
        """
        # Calculate number of samples from each dataset
        samples_per_dataset = {
            name: int(prop * output_size) 
            for name, prop in self._proportions.items()
        }
        
        # Adjust for rounding errors to ensure total is exactly output_size
        total = sum(samples_per_dataset.values())
        if total != output_size:
            diff = output_size - total
            # Add the difference to the largest proportion
            largest_dataset = max(self._proportions.items(), key=lambda x: x[1])[0]
            samples_per_dataset[largest_dataset] += diff
            
        # Sample from each dataset according to proportions
        mixed_samples = []
        for dataset in datasets:
            num_samples = samples_per_dataset[dataset.name]
            # Randomly sample without replacement
            sampled_indices = self._rng.choice(
                len(dataset.samples), 
                size=num_samples, 
                replace=True
            )
            mixed_samples.extend([dataset.samples[i] for i in sampled_indices])
            
        # Shuffle all samples together
        self._rng.shuffle(mixed_samples)
            
        # Create new mixed dataset with sampled data
        mixed_dataset = Dataset(
            name="mixed_dataset",
            size=output_size,
            samples=mixed_samples
        )
        return mixed_dataset