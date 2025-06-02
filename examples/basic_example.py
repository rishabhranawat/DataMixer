from core.dataset import Dataset
from algorithms.unimax import UniMax
from core.sampler import DataMixer

def generate_dummy_samples(size: int) -> list:
    """Generate dummy text samples for testing."""
    import random
    import string
    
    words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 
             'fig', 'grape', 'honeydew', 'kiwi', 'lemon']
    
    samples = []
    for _ in range(size):
        # Generate a random sentence with 3-7 words
        num_words = random.randint(3, 7)
        sentence = ' '.join(random.choices(words, k=num_words))
        samples.append(sentence)
    
    return samples

def main():
    # Create datasets with dummy samples
    dataset1 = Dataset(name="dataset1", size=100, samples=generate_dummy_samples(1000))
    dataset2 = Dataset(name="dataset2", size=2000, samples=generate_dummy_samples(2000))
    dataset3 = Dataset(name="dataset3", size=3000, samples=generate_dummy_samples(3000))
    
    # Create mixer
    mixer = DataMixer(
        datasets=[dataset1, dataset2, dataset3], 
        algorithm=UniMax(
            budget=6000, 
            seed=42, 
            max_epochs_per_language=2))
    mixer.mix()

    print(mixer.proportions)

    mixed_dataset = mixer.sample(datasets=[dataset1, dataset2, dataset3], output_size=6000)
    print(len(mixed_dataset.samples))

if __name__ == "__main__":
    main()