# MultiLayerPerceptron-CIFAR10

## Usage without Docker

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
venv/bin/python Train.py --num_epochs <num_epochs> --learning_rate <learning_rate> --optimizer <optimizer> --batch_size <batch_size>
```
- `<num_epochs>`: Number of epochs for training (default: 5)
- `<learning_rate>`: Learning rate for the optimizer (default: 0.01)
- `<optimizer>`: Optimizer to use (choices: `SGD`, `Adam`, default: `SGD`)
- `<batch_size>`: Batch size for training (default: 64)

### Test
```bash
venv/bin/python Test.py --model_path <model_path> --batch_size <batch_size>
```
- `<model_path>`: Path to the trained model file
- `<batch_size>`: Batch size for testing (default: 64)

## Usage using Docker

### Build the Docker image
```bash
docker build -t mlp_docker_image .
```

### Train
```bash
docker run -it --rm mlp_docker_image python Train.py --num_epochs <num_epochs> --learning_rate <learning_rate> --optimizer <optimizer> --batch_size <batch_size>
```

### Test
```bash
docker run -it --rm mlp_docker_image python Test.py --model_path <model_path> --batch_size <batch_size>
```