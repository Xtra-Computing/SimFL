This is the code of paper [Practical Federated Gradient Boosting Decision Trees](https://arxiv.org/pdf/1911.04206.pdf). The implementation is based on a previous version of [ThunderGBM](https://github.com/Xtra-Computing/thundergbm.git).

# Installation

Build SimFL:
```
git submodule init
git submodule update
mkdir build
cd build
cmake ..
make -j
```


# Usage

SimFL currently only works for binary classification tasks with labels 0 and 1 and requires GPUs.

## Prerequisites
* CMake
* CUDA

## Parameteres

```
* -p: int, number of parties (default:2)
* -t: int, number of lsh tables (default:40)
* -b: int, number of buckets (default:500)
* -r: float, r value of LSH function (default:4.0)
* -s: int, init seed for LSH
* -f: string, path to the dataset file
* -d: int, the maximum dimension of the datasets
```

## Datasets

Please rename all the local datasets in such format: name+'_train'+party_id, e.g., `a9a_train0`, `a9a_train1`. For the test dataset, please rename it in such format: name+'_test', e.g., `a9a_test`.

## Sample command:
Under `build` directory

```
./src/test/thundergbm-test -p 2 -t 30 -b 500 -r 4 -s -1 -f ../dataset/a9a/a9a -d 123 -n 50 -e 8
```

