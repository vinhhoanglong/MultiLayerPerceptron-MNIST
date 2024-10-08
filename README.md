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
venv/bin/python Train.py --num_epochs <num_epochs> --learning_rate <learning_rate> 
```
### Test
```bash
venv/bin/python Test.py --model_path <model_path> 
```


## Usage using docker


```bash
docker build -t mlp_docker_image .
```
Train
```bash
docker run -it --rm mlp_docker_image
```
Test
```bash
docker run -it --rm mlp_docker_image python Test.py
```