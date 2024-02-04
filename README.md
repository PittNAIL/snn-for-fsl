# snn-for-fsl

Code for the Paper "Few-Shot Learning for Clinical Natural Language Processing Using Siamese Neural
Networks: Algorithm Development and Validation Study"

Published at [JMIR AI][doi]

## Programs

- [`gpt2.py`](gpt2.py): GPT-2
- [`pt_snn.py`](pt_snn.py): Pre-trained Siamese Neural Network (PT-SNN)
- [`soe_snn.py`](soe_snn.py): Siamese Neural Network with Second-Order Embeddings (SOE-SNN)
- [`util.py`](util.py): A collection of utility functions

## Model Evaluation

```console
cd eval
./eval_pt_snn path_to_dataset.csv
./eval_soe_snn path_to_dataset.csv
./eval_gpt2.py
```

## Implementation

[snn-for-fsl][snn-for-fsl] has been implemented by [David Oniani][david].

[snn-for-fsl]: https://github.com/oniani/snn-for-fsl
[david]: https://oniani.ai
[doi]: https://doi.org/10.2196/44293
