# Context Value Informed ICRL

## Instructions for Setting Up the Environment


To create a new conda environment, open your terminal and run the following command:

```bash
conda create --name icrl python=3.9.15
```

Install PyTorch by following the [official instructions here](https://pytorch.org/get-started/locally/) appropriately for your system. 

```bash
conda install pytorch=1.13.0 torchvision=0.14.0 cudatoolkit=11.7 -c pytorch -c nvidia
```

The remaining requirements are fairly standard and are listed in the `requirements.txt`. These can be installed by running

```bash
pip install -r requirements.txt
```

Install Minigrid and stable-baselines3

```bash
cd Minigrid
pip install -e .
cd ..
```

```bash
cd stable-baselines3
pip install -e .
cd ..
```

