# MultiLayerPerceptron-MNIST

## Usage without docker

### Create virtual environment
```bash
python -m venv venv

```

### Install dependencies
```bash
venv/bin/pip install -r requirements.txt
```

### Train
```bash
venv/bin/python Train.py --num_epochs <num_epochs> (default: 5)
```
### Test
```bash
venv/bin/python Test.py --model_path <model_path> (default: model/mlp_20241006_220807.save)
```


## Usage using docker
this docker file is for model training step. You should modify it for testing step.

```bash
docker build -t mlp_docker_image .
```
```bash
docker run -it --rm mlp_docker_image
```