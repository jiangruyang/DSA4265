# read the acra csv file
import pandas as pd

acra_df = pd.read_csv('data/categorization/ACRA.csv')

# get the unique values of the entity_status_description column
statuses = acra_df['entity_status_description'].unique()

for status in statuses:
    print(status)
