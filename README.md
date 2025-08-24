# Mental Health Prediction Project

This project develops an end-to-end machine learning pipeline to predict the presence of mental illness in patients based on their demographic and clinical characteristics. 
Using the New York State Office of Mental Health Patient Characteristics Survey (PCS) 2019 dataset, we build predictive models and deploy them through a web application interface.

## Overview
<!-- Brief description of what the project doese -->

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
- **Age Groups**: Below 17 (Child), 18 and above (Adult), Unknown age
- **Race**: White only, Black Only, Multi-racial, Other, Unknown race
- **Ethnicity**: Non-Hispanic, Hispanic, Client Did Not Answer, Unknown

**Geographic & Service Variables:**
- **OMH Regions**: Provider location regions across New York State
- **Program Types**: Different mental health service programs
- **Service Utilization**: Various metrics related to mental health service usage

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
│  ├── data/
|  |  ├── models/
|  |    └── random_forest.pkl
|  ├── static/
|  |  ├── css/
|  |    └── style.css
|  ├── templates/
|  |  ├── base.html
|  |  ├── form.html
|  |  ├── index.html
|  |  └── result.html
|  └── app.py
├── data/
|  ├── models/
|  ├── processed/
|  ├── raw/
|    └── Patient.csv
├──  requirements.txt
└── README.md
```

## Usage
<!-- How to run the script -->

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
