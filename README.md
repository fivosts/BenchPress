<p align="center">
<img src="https://github.com/fivosts/clgen/blob/master/docs/logo.png" width="600px" />
</p>

***

:orange_book:  __BenchPress: A Deep Active Benchmark Generator__, *PACT 2022*.

__BenchPress__ is a directed program synthesizer for compiler benchmarks. Using active learning, it ranks compiler features based on their significance and produces executables that target them. __BenchPress__ is very accurate in generating compilable code - 87% of single-shot samples from empty feeds compile - while remaining light-weight compared to other massive seq2seq generative models.

## Quick Start

### Run a pre-trained model

You want to see some __BenchPress__ samples fast ? You can fetch and run a pre-trained model and experiment with your own string prompts.

```
export BENCHPRESS_BINARY=cmd
./benchpress

>>> from deeplearning.benchpress.models.from_pretrained import PreTrainedModel
>>> pretrained = PreTrainedModel.FromID("base_opencl")
>>> samples = pretrained.Sample("kernel void [HOLE]}")
>>> help(pretrained.Sample) # To list all Sample parameters.
```

### Training a model

After installing __BenchPress__ you can train your own model on a preset of C/OpenCL corpuses or using your own corpus. See all different model flavors in `model_zoo/BERT/`. To train, run

```
./benchpress --config model_zoo/BERT/online.pbtxt --workspace_dir <your_workspace_directory>
```

__BenchPress__ supports CPU, single node - single GPU, single node - multi GPU and multi node - multi GPU training and inference.

To see all available flags run `./benchpress --help/--helpfull`. Some relevant flags may be:

- `--sample_per_epoch` Set test sampling rounds per epoch end.
- `--validate_per_epoch` Similar to previous.
- `--local_filesystem` Set desired temporary directory. By default `/tmp/` is set.

## Installation

See `INSTALL.md` for instructions.
