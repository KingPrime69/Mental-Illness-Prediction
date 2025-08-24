# Mental Health Prediction Project

This project develops an end-to-end machine learning pipeline to predict the presence of mental illness in patients based on their demographic and clinical characteristics. 
Using the New York State Office of Mental Health Patient Characteristics Survey (PCS) 2019 dataset, we build predictive models and deploy them through a web application interface.

## Overview
The objective of this project is to develop a machine learning model pipeline that allows a user to determine whether they are likely to have a mental illness by answering a series of questions.
This project is divided into three main parts: Data Analysis, the creation and training of a prediction model for the presence or absence of Mental Illness, and finally the development of an interface that allows a patient to fill out a form and receive feedback on whether or not they are likely to have Mental Illness.
## Dataset Description

This project uses the **Patient Characteristics Survey (PCS) 2019** dataset from the New York State Office of Mental Health, available through the U.S. Government's open data catalog.

**Dataset Overview:**
- **Source**: [U.S. Government Data Catalog](https://catalog.data.gov/dataset/patient-characteristics-survey-pcs-2019)
- **Size**: 76 columns × 196,102 rows
- **Data Type**: Raw, uncleaned patient demographic and service data
- **Data Dictionary**: Available in `NYSOMH_PCS2019_DataDictionary` file
- **Target Variable**: Mental Illness column (presence/absence of mental illness)

### Data Features

The dataset is organized by OMH Region-specific (Region of Provider) and program type, with the following key demographic characteristics:

**Demographic Variables:**
- **Sex**: Male, Female, Unknown
- **Transgender Status**: No/Not Transgender, Yes/Transgender, Unknown  
- **Age Groups**: Below 21 (Child), 21 and above (Adult), Unknown age
- **Race**: White only, Black Only, Multi-racial, Other, Unknown race
- **Ethnicity**: Non-Hispanic, Hispanic, Client Did Not Answer, Unknown

**Geographic & Service Variables:**
- **OMH Regions**: Provider location regions across New York State
- **Program Types**: Different mental health service programs
- **Service Utilization**: Various metrics related to mental health service usage

**Health:**
-Various data about different diseases like Cancer

**Behaviour about Substance:**
-**Cannabis Use**: YES/NO
-**Alcohol Use**: YES/NO

**Data Quality Notes:**
- Contains significant missing data marked as "Unknown" categories
- Requires extensive preprocessing and cleaning
- Raw format necessitates careful handling of categorical variables

## Installation & Setup

### Prerequisites
<!-- Python version, required libraries -->
- Python : 3.11.9
- Pandas : 2.3.2
- Numpy : 2.2.6
- Matplotlib : 3.10.5
- Seaborn : 0.13.2
- XGBoost : 3.0.4
- Scikit-learn : 1.7.1
- Imbalanced-learn : 0.14.0
- SHAP : 0.48.0
- Flask : 3.1.2
- Joblib : 1.5.1

### Installation Steps
<!-- How to clone repo and install dependencies -->

```bash
# Clone the repository
git clone https://github.com/KingPrime69/Mental-Illness-Prediction.git
cd [Mental-Illness-Prediction]

# Install required packages
pip install -r requirements.txt
```

## Project Structure
<!-- Describe the Jupyter notebooks -->
```
├── app/
|  ├── static/
|  |  ├── css/
|  |    └── style.css
|  ├── templates/
|  |  ├── base.html
|  |  ├── form.html
|  |  ├── index.html
|     └── result.html
├── data/
|  ├── models/
|  |  ├── random_forest_model.pkl
|  |  └── xgb_model.pkl
|  ├── processed/
|  ├── raw/
|    └── Patient.csv
├── src/
|  ├── EDA.py
|  ├── preprocessing.py
|  └── training.py
├──  app.py
├──  requirements.txt
└── README.md
```

## Usage
**Data Analyse part** : First of all, the data must be cleaned and some columns are grouped together.
The while loops will prompt you to select the category of variables you wish to explore in order to display the corresponding graphs.
Once the data preprocessing and encoding are completed, a correlation analysis is performed to evaluate both the relationships and the strength of the associations between the different variables.
### Runnning the App
```bash
# Go in app folder
cd app

#run app.py
python app.py
```
When the server is up go to (http://127.0.0.1:5000)

### Running the Analysis

## Results
<!-- Model performance metrics, key findings -->

### Model Performance
<!-- Tables or charts showing accuracy, precision -->

### Key Insights
<!-- Main discoveries from the analysis -->

## Methodology
<!-- Brief explanation of algorithms used and approach -->

⚠️ **Important Notice**: This project is for educational purposes only and should not be used for actual clinical decision-making or patient diagnosis.

## Limitations
<!-- Discuss data limitations, model biases, generalizability issues -->


## Team Members
Antoine Petit
Maud Busserolles
Quentin Ripot

## License
<!-- Specify the license -->

---
*This project was developed as part of Machine Learning with Python Labs at [DSTI](https://dsti.school/fr)*
