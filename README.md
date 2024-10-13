<a id="readme-top"></a>

![GitHub repo size](https://img.shields.io/github/repo-size/krishujeniya/Fitness-Tracker-ML)
![GitHub contributors](https://img.shields.io/github/contributors/krishujeniya/Fitness-Tracker-ML)
![GitHub stars](https://img.shields.io/github/stars/krishujeniya/Fitness-Tracker-ML?style=social)
![GitHub forks](https://img.shields.io/github/forks/krishujeniya/Fitness-Tracker-ML?style=social)

# FitnessTracker

This project tracks exercise activities using sensors (e.g., accelerometer and gyroscope) and applies machine learning to classify and analyze the exercises. The project follows MLOps practices to ensure robust development, deployment, and maintenance.

## Table of Contents

1. [About The Project](#about-the-project)
   - [Key Features](#key-features)
   - [Project Structure](#project-structure)
2. [MLOps Integration](#mlops-integration)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Model Training and Inference](#model-training-and-inference)
   - [Train a Model](#train-a-model)
   - [Predict Using a Model](#predict-using-a-model)
5. [Docker Integration](#docker-integration)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

## About The Project

FitnessTracker uses sensor data to track and classify different exercises. This project organizes the entire machine learning pipeline from data acquisition, feature engineering, model training, and evaluation, to model deployment, following a well-structured MLOps approach.

### Key Features

- **Data Management**: Organizes raw, processed, and external data in a structured way.
- **Feature Engineering**: Transforms raw sensor data into meaningful features for modeling.
- **Modeling**: Trains machine learning models to classify exercises.
- **MLOps Pipeline**: Follows the best MLOps practices for scalable and maintainable ML development.
- **Docker Integration**: Containerizes the entire pipeline for consistent and reproducible environments.

## Project Structure

```plaintext
FitnessTracker
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- This README file
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- Final data sets for modeling
│   └── raw            <- Original sensor data
├── docs               <- Sphinx documentation
├── models             <- Trained models and summaries
├── notebooks          <- Jupyter notebooks for experimentation
├── references         <- Manuals, data dictionaries, etc.
├── reports            <- Generated reports, including figures and analysis
├── requirements.txt   <- Python dependencies
├── setup.py           <- Makes the project pip installable
├── src
│   ├── __init__.py    <- Initializes the src module
│   ├── data           <- Scripts for data processing and cleaning
│   ├── features       <- Scripts for feature engineering
│   ├── models         <- Scripts for model training and prediction
│   └── visualization  <- Scripts for data visualization
└── tox.ini            <- Settings for Tox to automate testing
```

## MLOps Integration

This project follows MLOps practices to streamline the workflow and ensure robust deployment of machine learning models. Key aspects include:

- **Data Versioning**: Each stage of data is versioned and stored in a structured way to maintain data consistency and reproducibility.
- **CI/CD Pipeline**: Automation of testing, model training, and deployment using tools like GitHub Actions or Jenkins.
- **Model Versioning**: Models are versioned and stored using DVC or MLflow for easy tracking.
- **Containerization**: Docker ensures a consistent environment across development, testing, and production stages.
  
## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

- **Python 3.x**: Install the latest version of Python.
- **Docker**: Ensure Docker is installed for containerization.

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/krishujeniya/FitnessTracker.git
   cd FitnessTracker
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up the environment:
   ```sh
   make setup
   ```

## Model Training and Inference

### Train a Model

To train the machine learning model using the preprocessed data:

```sh
make train
```

This will execute the training pipeline, including data loading, feature engineering, and model training, and save the trained model in the `models/` directory.

### Predict Using a Model

To make predictions using the trained model:

```sh
make predict
```

This script will load the model and apply it to the new input data, generating predictions that can be used for exercise classification.

## Docker Integration

The project can be containerized for consistent and reproducible execution across environments.

### Build the Docker Image

To build the Docker image:

```sh
docker build -t fitness-tracker .
```

### Run the Docker Container

To run the containerized application:

```sh
docker run -it --rm fitness-tracker
```

This will start the application within a Docker container, ensuring it runs in an isolated, stable environment.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Docker](https://www.docker.com/)
- [Open Source Community](https://opensource.org/)
- [Contributors](https://github.com/krishujeniya/FitnessTracker/graphs/contributors)
