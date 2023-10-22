# Disaster Response Project

## Table of Contents

  1 - Introduction\
  2 - Files Description\
  3 - Installation\
  4 - Instructions\
  5 - Acknowledgements\
  6 - Screenshots

## 1 - Introduction
This project is part of the Udacity's Data Scientist Nanodegree Program in collaboration with Appen (former Figure Eight).

The purpose of this project is to use disaster messages that were pre-labeled, to create a disaster response model to categorize messages received in real time during a "disaster event", and then these message could be sent to the specific disaster response agency.

A web application was built where the user can input messages received and get the classification for it.

## 2 - Files Description
### Folder: app
run.py - python script to launch web application.
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

### Folder: data
disaster_messages.csv - real messages sent during disaster events (provided by Appen)
disaster_categories.csv - categories of the messages
process_data.py - ETL pipeline used to load, clean, extract feature and store data in SQLite database
ETL Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ETL pipeline
DisasterResponse.db - cleaned data stored in SQlite database

### Folder: models
train_classifier.py - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use
classifier.pkl - pickle file contains trained model
ML Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ML pipeline

## 3 - Installation
There should be no extra libraries required to install apart from those coming together with Anaconda distribution. There should be no issue to run the codes using Python 3.5 and above.

## 4 - Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

## 5 - Acknowledgements
Udacity for providing an excellent Data Scientist training program. Figure Eight for providing dataset to train our model.

## 6 - Screenshots

Below it can be seen the Web App developed for this project. Some screenshots of the program can be verified on the "Screenshots" folder.

![Screenshot_WebApp](https://github.com/kcirne25/Disaster-response-pipeline/blob/main/Screenshots/Screenshot_WebApp.png)

