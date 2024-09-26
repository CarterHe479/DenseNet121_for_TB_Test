# DenseNet Image Classification with Custom Preprocessing

This repository contains the implementation of a DenseNet model for binary image classification with custom image preprocessing. The project includes model training, evaluation, and utilities to handle image preprocessing and prediction saving. The project is structured to be reusable, easy to navigate, and scalable.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Tuberculosis (TB) is one of the leading causes of death worldwide. Since there is no effective TB vaccine for adults, case identification and treatment are the main methods to control transmission. Mycobacterium tuberculosis can be efficiently identified in tissue sections by color and morphology through antacid staining, which opens up the possibility of pre-diagnosing tuberculosis using machine learning methods.

This project focuses on applying a DenseNet model for image classification, leveraging custom preprocessing methods such as grayscale conversion and template matching with rotation. The aim is to classify images effectively using a state-of-the-art convolutional neural network architecture.

The dataset used in this project can be customized, and the model can be trained on any grayscale image dataset for binary classification. 

## Features
- DenseNet model implemented using Keras and TensorFlow.
- Custom preprocessing pipeline for loading, filtering, and rotating images.
- Training and evaluation scripts with ROC curve plotting.
- Utility functions for saving predictions to CSV.

## Installation

To run this project locally, you need Python 3.7 or higher and `pip`. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
