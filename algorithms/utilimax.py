from core.base import DataMixingAlgorithm
from core.dataset import Dataset
from typing import Dict, List, Optional, Any

import cvxpy as cp
import numpy as np

class UtiliMax(DataMixingAlgorithm):
    """UtiliMax is a data mixing algorithm that mixes datasets according to the UtiliMax algorithm."""

    def __init__(self, 
                 budget: int, 
                 seed: Optional[int] = None,
                 max_epochs_per_language: Optional[int] = None):
        super().__init__(budget, seed)
        self.max_epochs_per_language = max_epochs_per_language

    def compute_proportions(self, 
                            datasets: List[Dataset],
                            U: np.array,
                            ) -> Dict[str, float]:
        """Compute the proportions of each dataset in the mixed dataset."""
        # Get dimensions for optimization problem
        num_tasks = U.shape[1]  # Number of utility dimensions
        num_datasets = len(datasets)  # Number of datasets to mix

        # Target utility vector (equal utility across all tasks)
        ones_vector_tasks = np.ones(num_tasks)

        # Extract dataset sizes for constraint calculation
        dataset_sizes = np.array([dataset.size for dataset in datasets])

        # Optimization variable: mixing weights for each dataset
        w = cp.Variable(num_datasets)

        # Minimize distance from target utility vector
        utility_distance = cp.norm(w @ U - ones_vector_tasks, 2)

        # Regularization term to prevent extreme weights
        risk_term = num_datasets * cp.sum_squares(w)

        # Combined objective: balance utility and risk
        objective = cp.Minimize(utility_distance + risk_term)

        # Constraints:
        # 1. Non-negative weights
        # 2. Weights sum to 1
        # 3. Respect max epochs per dataset
        constraints = [
            w >= 0, 
            cp.sum(w) == 1,
            (cp.multiply(self.budget, w) / dataset_sizes) <= self.max_epochs_per_language
        ]

        # Solve the convex optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Return dataset proportions if optimal solution found
        if problem.status == cp.OPTIMAL:
            return {datasets[i].name: w.value[i] for i in range(num_datasets)}
        else:
            raise ValueError("Problem is not optimal")