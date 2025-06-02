Definitely — here's your updated **README** with the **basic example and its output** included for clarity:

---

## **Data Mixing Library**

A Python library for combining multiple datasets using customizable algorithms. It offers a flexible framework to control the contribution of each dataset in the final output.

### **Overview**

Built on a modular design, the library cleanly separates combination strategies from sampling logic. This enables easy integration of new methods while maintaining a consistent interface.

---

## **Basic Example**

```python
from core.dataset import Dataset
from algorithms.unimax import UniMax
from core.sampler import DataMixer

# ...

# Initialize mixer with UniMax algorithm
mixer = DataMixer(
    datasets=[dataset1, dataset2, dataset3],
    algorithm=UniMax(budget=6000, seed=42, max_epochs_per_language=2)
)
mixer.mix()

# Sample mixed dataset
mixed_dataset = mixer.sample(datasets=[dataset1, dataset2, dataset3], output_size=6000)

print(mixer.proportions)
# {'dataset3': 0.47619047619047616, 'dataset2': 0.47619047619047616, 'dataset1': 0.047619047619047616}

print(len(mixed_dataset.samples))
# 6000
```
