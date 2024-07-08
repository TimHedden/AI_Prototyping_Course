# AI_Prototyping_Course

This repository contains projects for the AI Prototyping course. Applied knowledge from/in the Prototyping course:

### Skills applied:

1. Project 1: Used Streamlit interface, createt interactive Plotly graphs, used pre-trained model (vaderSentiment.SentimentIntensityAnalyzer)
2. Project 2: Used Dash interface, created interactive Plotly graphs, trained own ML model (rf_model.pkl), Used RAG (for recommender answers)
3. Project 3: Used Streamlit interface, used RAG/pre-trained model (Langchain chatbot integration), used LLM's + promt engeneering (for PDF data extraction and transaction categorization) 

### Changes Summary:

1. Project 1: Incorporated design feedback and created embeddings for categorization

2. Project 2: Made adjustments to the nutrition prediction page to use RAG for recommendation based on real input data


## Project 1: Customer Feedback Analysis

### Description
This project involves analyzing customer feedback using a Streamlit app. The app processes a CSV file containing customer feedback and provides insights.

### Files
- `STREAMLIT_app_v1.py`: The app for the first submission.
- `STREAMLIT_app_v2.1.py`: The app with plotly graphs to fulfill the design feedback. 
- `STREAMLIT_app_v2.2.py`: The app with plotly graphs to fulfill the design feedback and embeddings for feedback categorization to incorporate the technical feedback. 
- `Data/CustomerFeedback.csv`: The dataset containing customer feedback.


## Project 2: Open Foodfact DASH Daschboard

### Description
This project should give usefull insights in the key nutrition metrics of products across Germany, France and Spain and food categories like Baverages and Snacks. It includes two main functionalities and tabs:

1. Food analysis: The main dashboard should make a usefull comparission across product categories and countries possible. 
2. Product recommendation: The "product recommender" tab will analyse the ingredients of the inserted food and gives a consumption recommendation (self trained ML model) based on those values. On top it will search the openfood data to give you several products in the same category but with a better nutrition score.

### Files
- `DASH_app.py`: The main application script.
- `Modeltraining&Cleaning.ipynb`: Data cleaning and Model training for the ML recommendation model.
- `DATA/`: Analysis Dataset for the Dashboard (food_data2_cleaned) and API extracted data for the cleaning and model training
- `rf_model.pkl`: Contains the saved machine learning model. The model was serialized using Python's pickle module.
- `scaler.pkl`: Contains the saved scaler object. 
- `assets/`: Contains a font file (Acumin-BdPro.otf) for typography, a CSS file (DASH_Assignment2.css) for styling a web project, and an image file (Open_Food_Facts_logo_2022.png)


## Project 3: FincanceGuru - A personal finance aggregator
This project should be usefull to magnage the own personal finances. From a technical perspective it should showcase how to utilize the openAI integration to solve problems in the personal banking environment. 

### Files
- `app_FinanceGuru.py`: The main application script.
- `presentation_FinanceGuru.py`: In class presentation to present our project.
- `DATA/`: Test documents to try the aggregator app


### How to Run
1. Install the required dependencies.
   ```sh
   pip install -r requirements.txt