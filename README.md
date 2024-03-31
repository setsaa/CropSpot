# CropSpot

CropSpot is a plant disease identification application developed by UTS Team Helldiver.

## Description

CropSpot is designed to help farmers identify plant diseases quickly and accurately. By analyzing images of plants, the application uses machine learning algorithms to identify the specific disease affecting the plant. This allows users to take appropriate measures to prevent further spread and damage.

Our program currently supports the following plants, with more to be added in the future:
- Tomato

## Features

- Image recognition: CropSpot uses advanced image recognition techniques to identify plant diseases based on uploaded images.
- Disease database: The application has a comprehensive database of common plant diseases, allowing for accurate identification and information retrieval.
- User-friendly interface: CropSpot provides a simple and intuitive user interface, making it easy for users to navigate and use the application.

## The model

For our model, we used a pre-trained ResNet50 model with transfer learning to classify plant diseases. The model was trained on a dataset consisting of 24,881 total images, 5,435 of which are of tomato plants, which contains images of various plant diseases. By fine-tuning the model on our dataset, we were able to achieve high accuracy in identifying plant diseases. The images were preprocessed using data augmentation techniques to improve the model's performance.


## Installation

To install CropSpot, follow these steps:

1. Clone the repository: `git clone https://github.com/UTSTeamHelldiver/CropSpot.git`
2. Install the required dependencies from `requirements.txt.
3. Start the application.

## Usage

To use CropSpot, follow these steps:

1. Launch the application.
2. Upload an image of the plant with the suspected disease.
3. Wait for the application to analyze the image and provide the disease identification.
4. View detailed information about the identified disease and recommended treatments.
