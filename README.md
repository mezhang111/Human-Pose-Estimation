# 3D Human Pose and Shape Estimation from RGB Images

## Getting Started
This code has been implemented and tested with python >= 3.7.

```shell
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

## Training
```shell
python scripts/main.py --cfg configs/default.yaml
```

This command starts training our proposed model using the hyperparameters defined in `configs/defualt.yaml` YAML file. 
You can find all the configurable hyperparameters in `hps_core/core/config.py`. Feel free to add more parameters if you
feel the need to. If you want to perform a quick sanity check before starting full training, you can run:

```shell
python scripts/main.py --cfg configs/default.yaml
```
## Evaluation
```shell
python scripts/main.py --cfg configs/default_test.yaml
```
