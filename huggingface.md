# upload your own dataset to Huggingface

```bash
# !pip install huggingface_hub
# !huggingface-cli login
```

```py
import pandas as pd
import datasets
from datasets import Dataset

finetuning_dataset = Dataset.from_pandas(pd.DataFrame(data=finetuning_dataset))
finetuning_dataset.push_to_hub(dataset_path_hf)
```
