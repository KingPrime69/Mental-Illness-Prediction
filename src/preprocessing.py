# preprocessing.py
import pandas as pd
import numpy as np

def preprocessing(file_path=None, data=None, mode="train"):

    print("Starting preprocessing")

    if file_path is not None:
        df = pd.read_csv(file_path, sep=";", header=0)
        if "Mental Illness" in df.columns:
            df_unknown_mi = df[df["Mental Illness"] == "UNKNOWN"]
            df = df[df["Mental Illness"] != "UNKNOWN"]
        drop_cols_raw = [
            "Survey Year", "Three Digit Residence Zip Code", "Number Of Hours Worked Each Week",
            "Special Education Services", "Serious Mental Illness",
            "Principal Diagnosis Class", "Additional Diagnosis Class"
        ]
        df = df.drop(columns=[c for c in drop_cols_raw if c in df.columns], errors="ignore")
    else:
        df = data.copy()

    df = df.replace([
        "UNKNOWN", "NOT APPLICABLE",
        "UNKNOWN EMPLOYMENT HOURS", "UNKNOWN EMPLOYMENT STATUS",
        "CLIENT DID NOT ANSWER"
    ], np.nan)

    df = df.replace({"YES": 1, "NO": 0})

    drop_cols_common = [
        "Unknown Insurance Coverage", "Unknown Chronic Med Condition",
        "No Insurance", "Medicaid Insurance", "Medicaid Managed Insurance"
    ]
    df = df.drop(columns=[c for c in drop_cols_common if c in df.columns], errors="ignore")

    if mode == "train":
        df = df.dropna()
        df = df.drop_duplicates()

    if "Transgender" in df.columns:
        Trans = {
            'NO, NOT TRANSGENDER': 0,
            'YES, TRANSGENDER': 1,
            "CLIENT DIDN'T ANSWER": np.nan,
            'UNKNOWN': np.nan
        }
        df['Transgender'] = df['Transgender'].map(Trans)

    if {"Hispanic Ethnicity", "Race"}.issubset(df.columns):
        df['Ethnicity'] = 'UNKNOWN RACE'
        df.loc[df['Hispanic Ethnicity'] == 'YES, HISPANIC/LATINO', 'Ethnicity'] = 'HISPANIC/LATINO'
        no_hispanic = df['Hispanic Ethnicity'] == 'NO, NOT HISPANIC/LATINO'
        df.loc[no_hispanic, 'Ethnicity'] = df.loc[no_hispanic, 'Race']
        df = df.drop(columns=['Race', 'Hispanic Ethnicity'])
        df = df.replace("UNKNOWN RACE", np.nan)

    if "Sexual Orientation" in df.columns:
        df["Sexual Orientation"] = df["Sexual Orientation"].replace({
            'LESBIAN OR GAY': 'OTHER',
            'BISEXUAL': 'OTHER'
        })

    if "Ethnicity" in df.columns:
        df["Ethnicity"] = df["Ethnicity"].replace({
            'OTHER': 'OTHER',
            'MULTI-RACIAL': 'OTHER'
        })

    if "Preferred Language" in df.columns:
        df["Preferred Language"] = df["Preferred Language"].replace({
            'SPANISH': "OTHER",
            'INDO-EUROPEAN': "OTHER",
            'AFRO-ASIATIC': "OTHER",
            'ASIAN AND PACIFIC ISLAND': "OTHER",
            'ALL OTHER LANGUAGES': "OTHER"
        })

    if "Living Situation" in df.columns:
        df["Living Situation"] = df["Living Situation"].replace({
            'OTHER LIVING SITUATION': "OTHER",
            'INSTITUTIONAL SETTING': "OTHER"
        })

    if "Education Status" in df.columns:
        df["Education Status"] = df["Education Status"].replace({
            'COLLEGE OR GRADUATE DEGREE': "BELOW",
            'SOME COLLEGE': "BELOW",
            'PRE-K TO FIFTH GRADE': "BELOW",
            'NO FORMAL EDUCATION': "BELOW"
        })

    ohe_cols = [
        'Program Category', 'Region Served', 'Age Group', 'Sex',
        'Religious Preference', 'Employment Status', 'Education Status', 'Ethnicity',
        'Sexual Orientation', 'Preferred Language', 'Living Situation',
        'Household Composition', 'Veteran Status'
    ]
    for col in ohe_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_processed = df
    for col in ohe_cols:
        dummies = pd.get_dummies(df_processed[col], prefix=col)
        df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)

    df = df_processed

    df = df.replace({True: 1, False: 0})

    if mode == "train":
        df = df.dropna()

    print("Data preprocessing complete.")
    return df
