# snn-for-fsl

## Programs

- [`gpt2.py`](gpt2.py): GPT-2
- [`pt_snn.py`](pt_snn.py): Pre-trained Siamese Neural Network (PT-SNN)
- [`soe_snn.py`](soe_snn.py): Siamese Neural Network with Second-Order Embeddings (SOE-SNN)
- [`util.py`](util.py): A collection of utility functions

## Model Evaluation

```console
$ cd eval
$ ./eval_pt_snn path_to_dataset.csv
$ ./eval_soe_snn path_to_dataset.csv
$ ./eval_gpt2.py
```

## Implementation

[snn-for-fsl][snn-for-fsl] has been implemented by [David Oniani][david].

## License

[MIT License][license]

[snn-for-fsl]: https://github.com/oniani/snn-for-fsl
[david]: https://oniani.ai
[license]: LICENSE

