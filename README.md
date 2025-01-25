# Docker Setup

## 1- Run the following docker build command:

```docker build -t adamkaldas/generator:all .```

## 2- Run the container through the following command:

```docker run -d -it --name generator --network host --gpus all -v "$(pwd)/<your results folder>:/app/result" -v "$(pwd)/<Your data folder>:/app/data" adamkaldas/generator:all```

### Fill in the result and data names accordingly

## 3- Modify the documentation text and the content.json files in the data folder according to your needs.

## 4- get into the docker container through the following command: 

```docker exec -it generator /bin/bash```

# Running the Models

## 1- load the models using the command:

```python3 -u load_models.py```

## 2- Run the required model using:

```python3 -u <Model>.py```

### Fill in the model name (Phi, Qwen, Hermes).

## 3- The result.txt file should be in your results folder outside of the container.
