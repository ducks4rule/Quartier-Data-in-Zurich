import streamlit as st
import pandas as pd
from typing import List
from functools import reduce

from data_preparation import NAMES_DFs



def load_dataframe(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/cleaned/{name}.csv")

if __name__ == "__main__":
    key_quartier_kreis = load_dataframe('key_quartier_kreis')
    mapping_key = key_quartier_kreis['Quartier'].to_dict()  # index: value
    reverse_mapping_key = {v: k for k, v in mapping_key.items()}  # value: index

# def prepare_df(name: str, key=reverse_mapping_key,encoding_type='label', verbose=False) -> pd.DataFrame:
#     # check if file name ends on .csv
#     if name.endswith('.csv'):
#         name = name[:-4]
#
#     df = load_dataframe(name)
#     # key for unkown quartier
#     unkown_quartier = len(key.keys())
#     df['Quartier'] = df['Quartier'].apply(lambda x: key.get(x, unkown_quartier))
#
#     if df.isna().any().any():
#         if verbose:
#             print('There are NaN values in the dataframe:')
#             print(df.isna().sum())
#         df.fillna(0, inplace=True)
#
#     if df.duplicated().any():
#         if verbose:
#             print('There are duplicate values in the dataframe:')
#             print(df.duplicated().sum())
#         
#     string_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).any()]
#     numeric_string_columns = []
#     for col in string_columns:
#         non_null = df[col]
#         mask = ~non_null.apply(lambda x: str(x).replace('.', '', 1).isdigit())
#         non_null[mask] = '0'
#         df[col] = non_null
#         if not non_null.empty and non_null.apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
#             # Convert to float if any value contains '.', else int
#             if non_null.str.contains('\.').any():
#                 df[col] = df[col].astype(float)
#             else:
#                 df[col] = df[col].astype(int)
#             numeric_string_columns.append(col)
#     # Update string_columns after conversion
#     string_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).any()]
#
#     if string_columns:
#         if verbose:
#             print('There are string values in the dataframe:')
#             print(df[string_columns].applymap(lambda x: isinstance(x, str)).sum())
#         if encoding_type == 'label':
#             for col in string_columns:
#                 df[col], _ = pd.factorize(df[col])
#         else:
#             df_one_hot = pd.get_dummies(df, columns=string_columns)
#             df = pd.concat([df, df_one_hot], axis=1)
#             df.drop(columns=string_columns, inplace=True)
#
#     # change flot64 to float32
#     for col in df.select_dtypes(include=['float64']).columns:
#         df[col] = df[col].apply(pd.to_numeric, downcast="float")
#     # change int to unsigned int
#     for col in df.select_dtypes(include=['int']).columns:
#         df[col] = df[col].apply(pd.to_numeric, downcast="unsigned")
#
#     x = 0
#     return df


def merge_dfs_per_year(dfs: List[pd.DataFrame], years=[2024], verbose=False) -> pd.DataFrame:
    for y in years:
        # check if year exists in the DataFrame
        for i, df in enumerate(dfs):
            if y not in df['Year'].unique():
                if verbose:
                    print(f'Year {y} not found in dataframe {i}')
                # add year to dataframe and fill with 0
                new_row = pd.DataFrame({'Year': [y]})
                df = pd.concat([df, new_row], ignore_index=True)
                df.fillna(0, inplace=True)
                 
            
    # only keep the years we want
    dfs = [df[df['Year'].isin(years)] for df in dfs]
    if len(dfs) > 1:
        dfs = reduce(lambda left, right: pd.merge(left, right, on=['Year','Quartier'], how='outer'), dfs)
    else:
        dfs = dfs[0]

        
    return dfs








if __name__ == "__main__":
    df1 = prepare_df(NAMES_DFs[3],encoding_type='label', verbose=False)
    df2 = prepare_df(NAMES_DFs[4],encoding_type='label', verbose=False)
    dfs = [df1, df2]
    years = [2020, 2021]
    merged_df = merge_dfs_per_year(dfs,years, verbose=True)
    print(merged_df.head())
