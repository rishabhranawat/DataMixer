import numpy as np
from core.dataset import Dataset
from core.sampler import DataMixer
from algorithms.utilimax import UtiliMax

# Create 3 synthetic datasets with different sizes
dataset1 = Dataset(name="dataset1", size=1000, samples=[1] * 1000)  # Dummy samples
dataset2 = Dataset(name="dataset2", size=2000, samples=[2] * 2000)
dataset3 = Dataset(name="dataset3", size=1500, samples=[3] * 1500)

# Define utility matrix U where each row represents a dataset and each column represents a task
# U[i,j] represents the utility of dataset i for task j
U = np.array([
    [0.8, 0.2],  # dataset1 is good for task1, poor for task2
    [0.3, 0.7],  # dataset2 is poor for task1, good for task2
    [0.5, 0.5]   # dataset3 is balanced for both tasks
])

# Initialize UtiliMax algorithm
budget = 1000  # Total number of samples to mix
max_epochs = 2  # Maximum number of epochs per dataset
utilimax = UtiliMax(budget=budget, max_epochs_per_language=max_epochs)

# Create data mixer
mixer = DataMixer(
    datasets=[dataset1, dataset2, dataset3],
    algorithm=utilimax
)

# Mix the datasets
mixer.mix(U=U)

# Print the computed proportions
print("Computed mixing proportions:")
for dataset_name, proportion in mixer.proportions.items():
    print(f"{dataset_name}: {proportion:.3f}")

# Sample from the mixed distribution
mixed_dataset = mixer.sample(
    datasets=[dataset1, dataset2, dataset3],
    output_size=budget
)

print(f"\nCreated mixed dataset with {mixed_dataset.size} samples")
