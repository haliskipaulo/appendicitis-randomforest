import numpy as np
import pandas as pd

data = pd.read_csv('app_data.csv', sep=';')

data.drop(columns=[
    'Gynecological_Findings', 'Enteritis', 'Meteorism', 'Coprostasis',
    'Ileus', 'Conglomerate_of_Bowel_Loops', 'Bowel_Wall_Thickening',
    'Lymph_Nodes_Location', 'Pathological_Lymph_Nodes', 'Abscess_Location',
    'Appendicular_Abscess', 'Surrounding_Tissue_Reaction', 'Perforation',
    'Perfusion', 'Appendicolith', 'Target_Sign', 'Appendix_Wall_Layers'
])

target = ['Diagnosis', 'Severity', 'Management']


