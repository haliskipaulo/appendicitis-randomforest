import numpy as np
import pandas as pd
from pickle import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('app_data.csv', sep=';')

data.drop(columns=[
    'Gynecological_Findings', 'Enteritis', 'Meteorism', 'Coprostasis',
    'Ileus', 'Conglomerate_of_Bowel_Loops', 'Bowel_Wall_Thickening',
    'Lymph_Nodes_Location', 'Pathological_Lymph_Nodes', 'Abscess_Location',
    'Appendicular_Abscess', 'Surrounding_Tissue_Reaction', 'Perforation',
    'Perfusion', 'Appendicolith', 'Target_Sign', 'Appendix_Wall_Layers'
])

target_values = ['Diagnosis', 'Severity', 'Management']

num_columns = data.select_dtypes(include=np.number).columns
for column in num_columns:
    if data[column].isnull().any():
        median_temp = data[column].median()
        data[column] = data[column].fillna(median_temp)

categoric_columns = data.select_dtypes(include=['object']).columns.tolist()

cat_columns = [column for column in categoric_columns if column not in target_values]

for column in cat_columns:
    if data[column].isnull().any():
        moda = data[column].mode()[0] 
        data[column].fillna(moda, inplace=True)

cat_values = data.select_dtypes(include=['object']).columns.tolist()

cat_values = [column for column in cat_values if column not in target_values]

data = pd.get_dummies(data, columns=cat_values)

bool_columns = data.select_dtypes(include='bool').columns
for column in bool_columns:
    data[column] = data[column].astype(int)

severity_model = RandomForestClassifier(n_estimators=100, random_state=42)
diagnosis_model = RandomForestClassifier(n_estimators=100, random_state=42)
management_model = RandomForestClassifier(n_estimators=100, random_state=42)
severity_label_encoder = LabelEncoder()
diagnosis_label_encoder = LabelEncoder()
management_label_encoder = LabelEncoder()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

data['SeverityEncoded'] = severity_label_encoder.fit_transform(data['Severity'])
data['DiagnosisEncoded'] = diagnosis_label_encoder.fit_transform(data['Diagnosis'])
data['ManagementEncoded'] = management_label_encoder.fit_transform(data['Management'])

# treinar severity
severity_X = data.drop(columns=['SeverityEncoded', 'Severity', 'Diagnosis', 'Management'])
severity_Y = data['SeverityEncoded']

severity_X_train, severity_X_test, severity_Y_train, severity_Y_test = train_test_split(severity_X, severity_Y, test_size=0.2, random_state=42)

severity_scores = cross_val_score(severity_model, severity_X, severity_Y, cv=kfold, scoring='accuracy')

severity_model.fit(severity_X_train, severity_Y_train)

# treinar diagnosis
diagnosis_X = data.drop(columns=['DiagnosisEncoded', 'Severity', 'Diagnosis', 'Management'])
diagnosis_Y = data['DiagnosisEncoded']

diagnosis_X_train, diagnosis_X_test, diagnosis_Y_train, diagnosis_Y_test = train_test_split(diagnosis_X, diagnosis_Y, test_size=0.2, random_state=42)

diagnosis_scores = cross_val_score(diagnosis_model, diagnosis_X, diagnosis_Y, cv=kfold, scoring='accuracy')

diagnosis_model.fit(diagnosis_X_train, diagnosis_Y_train)

# treinar management

management_X = data.drop(columns=['ManagementEncoded', 'Severity', 'Diagnosis', 'Management'])
management_Y = data['ManagementEncoded']

management_X_train, management_X_test, management_Y_train, management_Y_test = train_test_split(management_X, management_Y, test_size=0.2, random_state=42)

management_scores = cross_val_score(management_model, management_X, management_Y, cv=kfold, scoring='accuracy')
management_model.fit(management_X_train, management_Y_train)

dump(severity_model, open('models/severity.model', 'wb'))
dump(severity_label_encoder, open('models/severity_label_encoder.model', 'wb'))
dump(severity_X.columns.tolist(), open('models/severity_X.model', 'wb'))

dump(diagnosis_model, open('models/diagnosis.model', 'wb'))
dump(diagnosis_label_encoder, open('models/diagnosis_label_encoder.model', 'wb'))
dump(severity_X.columns.tolist(), open('models/diagnosis_X.model', 'wb'))

dump(management_model, open('models/management.model', 'wb'))
dump(management_label_encoder, open('models/management_label_encoder.model', 'wb'))
dump(severity_X.columns.tolist(), open('models/management_X.model', 'wb'))
