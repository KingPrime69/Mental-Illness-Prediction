#pip install prince

import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import prince
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency


file_path="C:/Users/Antoine/Desktop/ML projet - Copie/ML projet - Copie/Project 2/MLProject.csv"

df = pd.read_csv(file_path, sep=";", header=0)

df.head()

for col in df.columns:
    uniques = df[col].unique()
    print(f"\n Colonne : {col}")
    print(uniques)




#Traitement de données
#Suppression doublons :
df = df.drop_duplicates()

#Isolation du groupe unknow de mental illness car on s'intéresse à ça en particulier. ça va nous servir pour entrainer
Mental_UNKNOWN=df[df["Mental Illness"] == "UNKNOWN"] 
df = df[df["Mental Illness"] != "UNKNOWN"]


Trans={'NO, NOT TRANSGENDER' : 0,
       'YES, TRANSGENDER' : 1,
       "CLIENT DIDN'T ANSWER" : np.nan,
       'UNKNOWN' : np.nan
       }
df['Transgender']=df['Transgender'].map(Trans)


#LGBT={
     # 'STRAIGHT OR HETEROSEXUAL' : 'H',
      #'LESBIAN OR GAY' : 'LG',
      #'OTHER' : 'O',
      #'BISEXUAL' : 'B',
      #'CLIENT DID NOT ANSWER' : 'UNK',
     # 'UNKNOWN': 'UNK'
     # }
#df['Sexual Orientation']=df['Sexual Orientation'].map(LGBT)


#Fusion de colonne redondante
df['Ethnicity'] = 'UNKNOWN RACE'
df.loc[df['Hispanic Ethnicity'] == 'YES, HISPANIC/LATINO', 'Ethnicity'] = 'HISPANIC/LATINO'
no_hispanic = df['Hispanic Ethnicity'] == 'NO, NOT HISPANIC/LATINO'
df.loc[no_hispanic, 'Ethnicity'] = df.loc[no_hispanic, 'Race']
df = df.drop(columns=['Race', 'Hispanic Ethnicity'])

#Race={
#      'HISPANIC/LATINO' : 'H',
#      'WHITE ONLY' : 'W',
#      'UNKNOWN RACE' : 'UNK',
#      'OTHER' : 'O',
#      'BLACK ONLY' : 'B',
#      'MULTI-RACIAL' : 'MR'
#      }

#df['Ethnicity']=df['Ethnicity'].map(Race)




#Je vais transformer les Yes/No/Unknown

#Création d'un dico

Dict_Binary={
    "YES" : 1,
    "NO" : 0,
    "UNKNOWN" : np.nan,
    True : 1,
    False :0
    }

#Création de la fonction de conversion

def Yes_to_1(df):
    for y in df.columns:
        unique_value = df[y].unique()
        est_ok = True
        for val in unique_value:
            if val not in Dict_Binary:
                est_ok = False
                break
        if est_ok:
            df[y] = df[y].map(Dict_Binary)
    return df

#application
df2=Yes_to_1(df)

#On check
for col in df2.columns:
            uniques = df2[col].unique()
            print(f"\n Colonne : {col}")
            print(uniques)
            
df=df2




#Il y a des colonnes qui sont proches donc on va les réunir dans des grandes catégories ça pourra servir
categories_column = {
    "Context" :["Survey Year","Program Category","Region Served"
        ],
    "Demographic": [
        "Age Group", "Sex", "Transgender", "Sexual Orientation",
        "Ethnicity","Preferred Language"
        ],
    "Social_eco" : [  "Living Situation", "Household Composition","Employment Status", "Number Of Hours Worked Each Week", "Education Status",
        "Veteran Status", "Criminal Justice Status","Religious Preference"
    ],
    "Insurance_Social_Assistance": [
        "SSI Cash Assistance", "SSDI Cash Assistance", "Veterans Disability Benefits",
        "Veterans Cash Assistance", "Public Assistance Cash Program", "Other Cash Benefits",
        "Medicaid and Medicare Insurance", "No Insurance", "Medicaid Insurance",
        "Medicaid Managed Insurance", "Medicare Insurance", "Private Insurance",
        "Child Health Plus Insurance", "Other Insurance"
    ],
    "Health_diagnose": [
        "Mental Illness", "Serious Mental Illness", "Principal Diagnosis Class", "Additional Diagnosis Class",
        "Intellectual Disability", "Autism Spectrum", "Other Developmental Disability",
        "Alcohol Related Disorder", "Drug Substance Disorder", "Opioid Related Disorder",
        "Hyperlipidemia", "High Blood Pressure", "Diabetes", "Obesity",
        "Heart Attack", "Stroke", "Other Cardiac", "Pulmonary Asthma", "Alzheimer or Dementia",
        "Kidney Disease", "Liver Disease", "Endocrine Condition", "Neurological Condition",
        "Traumatic Brain Injury", "Joint Disease", "Cancer", "Other Chronic Med Condition",
        "No Chronic Med Condition",
        "Mobility Impairment Disorder", "Hearing Impairment", "Visual Impairment", "Speech Impairment"
    ],
    "Behaviour_Service": [
        "Cannabis Recreational Use", "Cannabis Medicinal Use", "Smokes",
        "Received Smoking Medication", "Received Smoking Counseling",
        "Alcohol 12m Service", "Opioid 12m Service", "Drug/Substance 12m Service"
    ]
}





#Partie descripitve

#Tronche des données





while True:
    print("\nAvailable variable groups:")
    for name in categories_column.keys():
        print("-", name)
    print("Type 'stop' to exit")
    grp_name = input("Enter the name of the variable group you want to select: ")

    if grp_name.lower() == "stop":
        print("Stopping all plots. Goodbye!")
        break

    elif grp_name in categories_column:
        grp = categories_column[grp_name]
        print(f"You selected: {grp}")
       
        # Loop to check each variable
        for c in grp:
            print(f"\nVariable : {col}")
            print(df[c].value_counts(dropna=False))
        
        
        
            plt.figure(figsize=(8,4))
            graph=sns.countplot(x=c, data=df, order=df[c].value_counts().index)
            plt.title(f"Distribution de {col}")
            plt.xticks(rotation=45, ha='right')
            
            tot = len(df)
            for x in graph.patches:
               height = x.get_height()
               p = height / tot * 100
               graph.text(x.get_x() + x.get_width()/2, height + 0.5, f"{int(p)}%", ha='center')
        
            plt.tight_layout()
            plt.show()
    
else:
    print("Invalid selection")
    
    
    #plt.figure(figsize=(10,5))
    #sns.countplot(x=col, hue='Mental Illness',data=df, order=df[col].value_counts().index)
    #plt.title(f"Distribution de {col} with Mental Illness")
    #plt.xticks(rotation=45, ha='right')
    #plt.tight_layout()
    #plt.show()
    



#Test de chi2 à part :
    
while True:
    print("\nAvailable variable groups:")
    for name in categories_column.keys():
        print("-", name)
    print("Type 'stop' to exit")
    grp_name = input("Enter the name of the variable group you want to select for Chi² test: ")

    if grp_name.lower() == "stop":
        print("Stopping Chi² tests. Goodbye!")
        break

    elif grp_name in categories_column:
        grp = categories_column[grp_name]
        print(f"You selected: {grp}")

        # Loop to check each variable
        for col in grp:
            table = pd.crosstab(df[col], df['Mental Illness'])

            try:
                chi2, p, dof, expected = chi2_contingency(table)
                print(f"\nVariable: {col}")
                print(f"Chi² = {chi2:.2f}, p-value = {p:.2e}, degrees of freedom = {dof}")
                if p < 0.05:
                    print(" Significant association between", col, "and Mental Illness ")
                else:
                    print(" Non-significant association between", col, "and Mental Illness")
            except Exception as e:
                print(f"Error performing Chi² for {col}: {e}")

    else:
        print("Invalid selection")




#ACp pour binaire : MCA
context_cols = categories_column["Health_diagnose"]
#remplace apr Nan
df_test = df[context_cols].replace('UNKNOWN', np.nan)

# df_mca = dataframe avec colonnes catégorielles ou déjà en one-hot
mca = prince.MCA(n_components=4, random_state=999)
mca = mca.fit(df_test)
inertia = mca.eigenvalues_

# Coordonnées des individus
X_mca = mca.transform(df_test)

# Visualisation
X_mca.plot.scatter(x=0, y=1, title='MCA (Analyse des correspondances multiples)')
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')



# 5. Coordonnées des modalités (projection des colonnes)
modalities = mca.column_coordinates(df_test)

# 6. Visualisation des modalités (style "cercle de corrélation")
plt.figure(figsize=(8,8))
plt.scatter(modalities[0], modalities[1], c='red')

for i, txt in enumerate(modalities.index):
    plt.annotate(txt, (modalities.iloc[i, 0], modalities.iloc[i, 1]),
                 xytext=(3,3), textcoords='offset points')

plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.title("MCA - Projection des modalités")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()



















#V2

for col in context_cols:
    print(f"\nVariable : {col}")
    print(df[col].value_counts(dropna=False))
       

    # Distribution simple
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution de {col}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Distribution empilée Mental Illness (Yes / No)
    crosstab = pd.crosstab(df[col], df['Mental Illness Label'])
    crosstab = crosstab.loc[df[col].value_counts().index]  # Pour garder le bon ordre

    crosstab.plot(kind='bar', stacked=True, figsize=(10,5), colormap='viridis')
    plt.title(f"Distribution empilée de {col} par Mental Illness")
    plt.xlabel(col)
    plt.ylabel("Nombre d'individus")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Mental Illness")
    plt.tight_layout()
    plt.show()

    # Tableau de contingence
    table = pd.crosstab(df[col], df['Mental Illness'])
    print("Tableau de contingence :")
    print(table)

    # Test du Chi²
    try:
        chi2, p, dof, expected = chi2_contingency(table)
        print(f"Chi² = {chi2:.4f}, p-value = {p:.4e}, degrés liberté = {dof}")
    except Exception as e:
        print(f"Erreur Chi² : {e}")








sns.displot(data=df,x="Ethnicity")
plt.show()

sns.displot(data=df,x="Number Of Hours Worked Each Week")
plt.xticks(rotation=90)
#plt.tight_layout()
plt.show()

for col in categories_column["Context"]:
    plt.figure()
    sns.displot(data=df, x=col)
    plt.title(col)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    
    
sns.countplot(data=df, x="Sex", hue="Mental Illness")



