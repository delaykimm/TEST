## Requirements

 - Numpy (`numpy`) v1.15.2;
 - Matplotlib (`matplotlib`) v3.0.0;
 - Orange (`Orange`) v3.18.0;
 - Pandas (`pandas`) v0.23.4;
 - `python-weka-wrapper3` v0.1.6 for multivariate time series (requires Oracle JDK 8 or OpenJDK 8);
 - PyTorch (`torch`) v0.4.1 with CUDA 9.0;
 - Scikit-learn (`sklearn`) v0.20.0;
 - Scipy (`scipy`) v1.1.0.
 - Huggingface (`transformers`)

## Datasets

The datasets manipulated in this code can be downloaded on the following locations:
 - the UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/;
 - the UEA archive: http://www.timeseriesclassification.com/;

## Files

### Core

 - `losses` folder: implements the triplet loss in the cases of a training set
   with all time series of the same length, and a training set with time series
   of unequal lengths;
 - `networks` folder: implements encoder and its building blocks (dilated
   convolutions, causal CNN);
 - `scikit_wrappers.py` file: implements classes inheriting Scikit-learn
   classifiers that wrap an encoder and an SVM classifier.
 - `llm_wrappers.py` file: implements classes that wrap an CNN encoder, LLM, and a classifier.
 - `default_hyperparameters.json` file: example of a JSON file containing the
   hyperparameters of a pair (encoder, classifier).

### Tests

 - `ucr.py` file: handles learning on the UCR archive (see usage below);
 - `uea.py` file: handles learning on the UEA archive (see usage below);

## Usage

### Selecting text prototype

Download LLM from huggingface

To select text prototypes from GPT2

`python losses/text_prototype.py --llm_model_dir= path/to/llm/folder/ --prototype_dir path/to/save/prototype/file/ --provide Flase(ramdom) or a text lisr --number_of_prototype 10`



### Training on the UCR and UEA archives

To train a model on the Mallat dataset from the UCR archive with specific gpu:

`python ucr.py --dataset Mallat --path path/to/Mallat/folder/ --prototype_file path/to/prototype/file --save_path /path/to/save/models --hyper default_hyperparameters.json --cuda --gpu 0`

To train a model on the Mallat dataset from the UCR archive with DDP:

`python -m torch.distributed.launch --nproc_per_node=8 ucr.py --dataset Mallat --path path/to/Mallat/folder/ --prototype_file path/to/prototype/file --save_path /path/to/save/models --hyper default_hyperparameters.json --cuda`

Adding the `--load` option allows to load a model from the specified save path.
Training on the UEA archive with `uea.py` is done in a similar way.

