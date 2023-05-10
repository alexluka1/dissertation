# Dissertatuion Project - Classifying Alzheimer's Disease into 4 Classes

## Repo Info
Contains code needed to complete dissertation at the university of Lincoln. This code can categorise MRI images of brains into differnt categories of Alzheimer's disease. It works using a Convolutional Neural Network adn uses data from the Kaggle datasets 
https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset?resource=download and https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images 


Project.ipynb - Notebook file containing code to run all of the custom models. 
TransferLearning.ipynb - Notebook file containing code to implement transfer learning models.
hyperparameters.ipynb - Notebook file containing code to fine tune hyperparameters and create new model through hyperparameter method
all_models.py - Simple Python file containing the code for all the custom models

## Useage Instructions 
To run all the code:
- Upload the data.zip file into Google Colab
- Wait for data to be uploaded
- Click on runtime and press runall

To change the custom model:
- Scroll to model creation code, this will be marked by a section called "Creating Sequential deep learning model"
- Copy the code for a model from all_models.py
- Delete code from block containing "model = keras.Sequential()"
- Paste new model where deleted code was 
- If the learning rate is specified with an lr variable in all_models.py, then change the learning rate in model.compile

To change the transfer learning model:
- Go to the creating model section
- Change the base model into the pre-built model wanted
- Do this by changing the line : "base_model = tf.keras.applications.resnet50.ResNet50" to another model. Such as "base_model = tf.keras.applications.EfficientNetB1"
- These different pre-built models can be found at https://www.tensorflow.org/api_docs/python/tf/keras/applications


