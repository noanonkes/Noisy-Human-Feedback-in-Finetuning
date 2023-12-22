# Fine-Tuning Models: The Influence of Noisy Human Feedback in Fine-tuning from Preferences
Welcome to the GitHub repository for the research project conducted at the University of Amsterdam, focusing on "Fine-Tuning Models: The Influence of Noisy Human Feedback in Fine-tuning from Preferences." 

## Overview
This study investigates the impact of noise introduction during fine-tuning on model performance. Leveraging insights from Reinforcement Learning with Human Feedback (RLHF), we implement a loss function enabling end-to-end fine-tuning. We systematically introduce controlled noise levels into datasets during fine-tuning and observe their effects on model performance. Our methodology involves initial training, fine-tuning with noise, and incrementally increasing noise levels for analysis. Findings reveal unexpected benefits from noise introduction, but extended fine-tuning results in performance decline, signalling sub-optimal loss function implementation. Future research may explore RLHF, reinforcement learning aspects, and regularization techniques to refine fine-tuning and enhance model performance.

## Environment and requirements

As the directory structure shows, we included a `environment.yml` file with the packages and dependencies that should be installed. You can create a virtual environment with your favourite manager, i.e. conda, and install the requirements with:

```bash
conda env create -f environment.yml
```

## Usage
This repository is divided in two folder: `multiclass` and `binary`. The `multiclass` folder has everything to work with $k\geq 1$ and `binary` has everything for the selective pair selection. 

### Multiclass

The main function which runs the training and fine-tuning for one seed. You can run the file by calling the following in the `multiclass`
 folder:
 
```bash
python main.py [arguments]
```

It has the following arguments:

```bash
--use-cuda: Flag to enable GPU acceleration if available.
--path: Path to the data directory.
--model_name: Name of the model used to embed sentences.
--pc: Flag to use pre-calculated embeddings.
--ratio: Specifies the amount of noise added to the ratings.
--k: Determines the number of comparisons to use during fine-tuning.
--epochs-t: Number of epochs for training.
--epochs-f: Number of epochs for fine-tuning.
```

To run the experiment for multiple seeds, multiple noise levels and create a plot, you can run the following file in the folder `multiclass`:

```bash
python config.py [arguments]
```

With the following arguments:
```bash
--use-cuda: Flag to enable GPU acceleration if available.
--path: Path to the data directory.
--model_name: Name of the model used to embed sentences.
--pc: Flag to use pre-calculated embeddings.
--ratio: Specifies the amount of noise added to the ratings.
--k: Determines the number of comparisons to use during fine-tuning.
--epochs-t: Number of epochs for training.
--epochs-f: Number of epochs for fine-tuning.
```

### Binary

The main function which runs the training and fine-tuning for one seed. You can run the file by calling the following in the `binary`
 folder:
 
```bash
python main.py [arguments]
```

It has the following arguments:

```bash
--use-cuda: Flag to enable GPU acceleration if available.
--path: Path to the data directory.
--model_name: Name of the model used to embed sentences.
--pc: Flag to use pre-calculated embeddings.
--ratio: Specifies the amount of noise added to the ratings.
--epochs-t: Number of epochs for training.
--epochs-f: Number of epochs for fine-tuning.
```

To run the experiment for multiple seeds, multiple noise levels and create a plot, you can run the following file in the folder `binary`:

```bash
python config.py [arguments]
```

With the following arguments:
```bash
--use-cuda: Flag to enable GPU acceleration if available.
--path: Path to the data directory.
--model_name: Name of the model used to embed sentences.
--pc: Flag to use pre-calculated embeddings.
--ratio: Specifies the amount of noise added to the ratings.
--epochs-t: Number of epochs for training.
--epochs-f: Number of epochs for fine-tuning.
```


## Authors

This research project is done by Noa Nonkes, from the University of Amsterdam.