import pandas as pd

### loading raw data and factors
df = pd.read_csv('/Users/ericliao/Desktop/metabolomics_workbench_studies/PR000058/ST000061_ST000065/data/aligned.csv')
df_factor = pd.read_csv('/Users/ericliao/Desktop/metabolomics_workbench_studies/PR000058/ST000061_ST000065/data/factors.csv')

# get quantitative data
df_height = df.loc[:, [' height' in i for i in df.columns]]
df_height = df_height.fillna(0)

column_names = []

for item in df_height.columns:
    local_sample_id = item.split(".", 1)[0]
    treatment_type = df_factor[df_factor['local_sample_id'] == local_sample_id]['Source'].values[0]
    new_column_name = local_sample_id + '_' + treatment_type

    column_names.append(new_column_name)

df_height.columns = column_names

df_height.to_csv("/Users/ericliao/PycharmProjects/Phd_Class/advance_stats/statsData_st000061_st000065.csv")
