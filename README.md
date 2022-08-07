<p align="center">
  <br>
<img src="https://github.com/fivosts/clgen/blob/master/docs/logo.png" width="600px" />
<br>
</p>

***

:orange_book:  __BenchPress: A Deep Active Benchmark Generator__, *PACT 2022*.

__BenchPress__ is a directed program synthesizer for compiler benchmarks. Using active learning, it ranks compiler features based on their significance and produces executables that target them. __BenchPress__ is very accurate in generating compilable code - 87% of single-shot samples from empty feeds compile - while remaining light-weight compared to other massive seq2seq generative models.

## Quick Start

### Run a pre-trained model

You want to see some __BenchPress__ samples fast ? You can fetch and run a pre-trained model and experiment with your own string prompts.

```
$: export BENCHPRESS_BINARY=cmd
$: ./benchpress

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

## Evaluate the code

If you have trained __BenchPress__ and ran a sampler to any downstream task you want to evaluate, you can use the codebase's evaluators. The evaluators usually take a list of database groups and perform operations/analysis/plotting on them. Evaluators are described in protobuf files (see examples in `model_zoo/evaluation/`). To run an evaluator run

```
$: export BENCHPRESS_BINARY=deeplearning/benchpress/experiments/evaluators.py
$: ./benchpress --evaluator_config <path/to/your/evaluator.pbxt>
```

## Github and BigQuery mining

__BenchPress__ provides modules to scrape source code from Github and store it into databases. Language specifications are set through protobuf files. See `model_zoo/github` for examples. For example

```
./benchpress --config model_zoo/github/bq_C_db.pbtxt
```
to scrape C repositories from BigQuery.

__BenchPress__ comes with two datasets. A dataset of ~64,000 OpenCL kernels and a C dataset of ~6.5 million source files (about ~90 million functions). The OpenCL dataset is downloaded automatically if requested through a model description protobuf (see corpus field). The C database doesn't due to its size. If you are interested in it, get in touch.

## Utilities

A range of useful ML utilities reside within __BenchPress's__ codebase that you may find useful. Inside `deeplearning/benchpress/util` you will find standalone modules such as:

- `plotter.py`: A plotly interface that easily plots lines, scatters, radars, bars, groupped bars, stacked bars, histograms, distributions etc.
- `distrib.py`: A utility module that handles distributed environments: barrier(), lock(), unlock(), broadcast_messages() etc.
- `memory.py` : A RAM and GPU memory live tracker and plotter of your application.
- `gpu.py`: Wrapper over `nvidia-smi` for GPU info.
- `monitors.py`: A set of classes that monitor streaming data, store and plot.
- `distributions.py`: Class for distribution operations. Populate distributions and do operations on them (+, -, /, *) and plot PMFs, PDFs.
- `logging.py`: logging module with pretty colors.
- and others!
