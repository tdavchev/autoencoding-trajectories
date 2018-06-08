# Dynamic Seq2Seq for Trajectory Understanding

Using Seq2Seq to assign actions to trajectories.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For this repo you need:

 - tensorflow -v 1.4.0
 - Numpy -v 1.14.2
 - Scikit-learn -v 0.19.1
 - argparse -v 1.1

```
pip install tensorflow-gpu=1.4.0
pip install numpy, scikit-learn, argparse
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Download the repository.

```
git clone git@github.com:yadrimz/autoencoding-trajectories.git
```

You can train and infer. 
Run help for full documentation.

```
python main.py -h
```

To train:

```
python main.py --mode 'train'
```

Sample output:

```
Batch: 1
  minibatch_loss: 1.0833992958068848
  accuracy: 0.2222222222222222
   sample: 1
      sequence real id         :> 8
      target start             :> [4 3 5]
      predicted start          :> [3 3 1]
      target end region        :> [3 5]
      predicted end region     :> [3 1]
   sample: 2
      sequence real id         :> 7
      target start             :> [3 2 5]
      predicted start          :> [6 5 5]
      target end region        :> [2 5]
      predicted end region     :> [5 5]
   sample: 3
      sequence real id         :> 14
      target start             :> [4 1 5]
      predicted start          :> [5 3 4]
      target end region        :> [1 5]
      predicted end region     :> [3 4]
```

To infer after having trained a model:

```
python main.py --mode 'infer'
```

### Project Structure

```
Project File Structure:
-data
---timeseries_25_May_2018_18_51_49-walk3.npy
-models
---model.py
-utils
---utils.py
-main.py
-README.md
```

## Authors

* **Todor Davchev**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Tensorflow's NMT Tutorial](https://github.com/tensorflow/nmt/tree/tf-1.4)
* [Ematviev's GitHub repo](https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb)