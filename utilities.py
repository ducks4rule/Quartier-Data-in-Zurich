import numpy as np
import pandas as pd
from typing import List
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from PIL import ImageFont


# Dictionary w/ coordinates for the quartiers
QUARTIER_COORDS = {
    'Affoltern': (300, 100),
    'Seebach': (425, 75),
    'Oerlikon': (500, 200),
    'Saatlen': (600, 150),
    'Schwamendingen-Mitte': (580, 230),
    'Hirzenbach': (675, 275),
    'Höngg': (250, 210),
    'Altstetten': (100, 300),
    'Wipkingen': (375, 275),
    'Unterstrass': (450, 300),
    'Oberstrass': (525, 325),
    'Fluntern': (600, 375),
    'Hottingen': (675, 425),
    'Hirslanden': (650, 500),
    'Witikon': (750, 550),
    'Weinegg': (600, 600),
    'Mühlebach': (550, 550),
    'Seefeld': (510, 580),
    'Hochschulen': (525, 480),
    'Rathaus': (500, 430),
    'Lindenhof': (460, 450),
    'City': (425, 475),
    'Enge': (410, 530),
    'Wollishofen': (410, 700),
    'Leimbach': (325, 750),
    'Friesenberg': (275, 575),
    'Alt-Wiedikon': (350, 525),
    'Sihlfeld': (300, 450),
    'Albisrieden': (175, 475),
    'Werd': (380, 450),
    'Langstrasse': (400, 410),
    'Gewerbeschule': (410, 360),
    'Escher Wyss': (325, 325),
    'Hard': (315, 380),
    'Unbekannt (Stadt Zürich)': (700, 750),
    'Nicht zuordenbar': (700, 750),
    'Ganze Stadt': (700, 750),
    'Kreis 1': (500, 430),
    'Kreis 2': (410, 700),
    'Kreis 3': (350, 525),
    'Kreis 4': (400, 410),
    'Kreis 5': (325, 325),
    'Kreis 6': (450, 300),
    'Kreis 7': (675, 425),
    'Kreis 8': (550, 550),
    'Kreis 9': (100, 300),
    'Kreis 10': (250, 210), 
    'Kreis 11':  (500, 200),
    'Kreis 12':  (580, 230),
}
    
def draw_circle(x, y, r, width=1, draw=None, outline="red", fill=(255,0,0,80)):
    assert draw is not None, "Draw object must be provided"

    left_up = (x - r, y - r)
    right_down = (x + r, y + r)
    draw.ellipse([left_up, right_down], outline=outline, width=width, fill=fill)

def draw_in_quartier(q: str, draw=None, r=1, width=1, outline="red", fill="orange", text=None, text_fill="black", text_size=12):
    assert draw is not None, "Draw object must be provided"
    assert q in QUARTIER_COORDS, f"Quartier {q} not found in coordinates dictionary"
    # if r is empty array, set to 0
    if isinstance(r, (list, np.ndarray)):
        if len(r) == 0:
            r = 0
        else:
            r = r[0]
    elif isinstance(r, (int, float)):
        pass
    else:
        raise ValueError("r must be an int, float, list or numpy array")
    
    x, y = QUARTIER_COORDS[q]
    draw_circle(x, y, r, width=width, draw=draw, outline=outline, fill=fill)
    if text is not None:
        if text_size is not None:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMNerdFont-Bold.ttf", text_size)
            # font = ImageFont.truetype(text_font.path, text_size)
        draw.text((x, y), str(text), fill=text_fill, font=font, anchor="mm")






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

    dfs.drop_duplicates(inplace=True)
        
    return dfs



def scale_columns_ignore_nan(df):
    return df.apply(lambda col: (col - col.mean(skipna=True)) / col.std(skipna=True) if col.std(skipna=True) != 0 else col - col.mean(skipna=True))




def create_dict_to_table(dict):
    max_len = max(len(features) for features in dict.values())
    columns = []
    for cl in dict.keys():
        columns.extend([f'Cluster {cl} Feature', f'Cluster {cl} Value'])

    rows = []
    for i in range(max_len):
        row = []
        for cl in dict.keys():
            if i < len(dict[cl]):
                feat, diff = dict[cl][i]
                row.extend([feat, f"{diff:.2f}"])
            else:
                row.extend(['', ''])
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)




def compute_trend_slopes(df, n_round=4):
    lr_model = LinearRegression()
    trends = pd.DataFrame(index=df.columns, columns=['Trend'])
    for i, col in enumerate(df.columns):
        y = df[col].dropna().values
        if len(y) < 2:
            trends.iloc[i, 0] = np.nan
            continue
        X = np.arange(len(y)).reshape(-1, 1)
        lr_model.fit(X, y)
        trends.iloc[i, 0] = round(lr_model.coef_[0], n_round)
    return trends




def compute_features_for_features(
    df,
    df_diff,
    window_width=5,
):
    # Standard deviation
    std_vec = df.std()
    std_diff = df_diff.std()
    std_sort = std_vec.sort_values(ascending=False)
    std_diff_sort = std_diff.sort_values(ascending=False)

    # Trend slopes
    trend_slope = compute_trend_slopes(
        scale_columns_ignore_nan(df)
    ).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)
    trend_slope_diff = compute_trend_slopes(
        scale_columns_ignore_nan(df_diff)
    ).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)

    # Rolling mean
    df_rolling = df.rolling(window_width).mean()
    df_rolling_slope = compute_trend_slopes(
        scale_columns_ignore_nan(df_rolling)
    ).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)

    # Autocorrelation
    df_autocorr = df.rolling(window_width).apply(lambda x: x.autocorr(), raw=False)

    # Feature summary DataFrame
    df_compare_tot = pd.DataFrame({
        'Feature': list(df.columns),
        'Std': list(std_sort.values),
        'Trend': trend_slope.values.ravel(),
        'Diff Std': list(std_diff_sort.values),
        'Diff Trend': trend_slope_diff.values.ravel(),
        'Mean': list(df.mean()),
        'Rolling Trend': df_rolling_slope.values.ravel(),
        'Rolling Std': list(df_rolling.std()),
        'Rolling Max Change': list(df.max() - df.min()),
        'Autocorr Max': list(df_autocorr.max()),
        'Autocorr Min': list(df_autocorr.min()),
        # 'Rolling Skew': list(df_numeric_tot.rolling(5).skew()),
        # 'Rolling Kurtosis': list(df_numeric_tot.rolling(5).kurt()),
    })
    return df_compare_tot



def clean_data(df, switch=True):
    if switch:
        # remove columns with all NaN values
        df = df.dropna(axis=1, how='all')
        # remove columns with only one unique value
        for col in df.columns:
            if df[col].nunique() == 1:
                print(f"Removing column {col} with only one unique value")
                df = df.drop(columns=[col])

        # remove duplicates
        df = df.drop_duplicates()

        # convert object columns to numeric if possible, else to category
        for col in df.select_dtypes(include=[object]).columns:
            col_non_null = df[col].dropna()
            # col_clean = col_non_null.str.replace('K', '.', regex=False)
            col_clean = col_non_null.str.replace('K', '0', regex=False)
            is_numeric = col_clean.apply(lambda x: x.replace('.', '', 1).isdigit() or x == '.').all()
            if is_numeric:
        # for col in df.select_dtypes(include=[object]).columns:
        #     if df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit() or str(x)== '.').all():
                mask = ~df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())
                df.loc[df[col].index[mask], col] = '0'
                # Convert to float if any value contains '.', else int
                if df[col].str.contains('.').any():
                    df[col] = df[col].astype(float)
                else:
                    df[col] = df[col].astype(int)
            else:
                # convert to category
                df[col] = df[col].astype('category')

        # downcasting of numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == np.int64:
                # df[col] = pd.to_numeric(df[col], downcast='integer')
                df[col] = pd.to_numeric(df[col], downcast='unsigned')
            elif df[col].dtype == np.float64:
                df[col] = pd.to_numeric(df[col], downcast='float')

        df.select_dtypes(include=[np.number]).fillna(0, inplace=True)

    return df




@st.cache_data
def load_dataframe():
    df = pd.read_csv('data/cleaned/q_merged_data_small_5.csv')
    return clean_data(df, switch=True)
    # return pd.read_csv('data/cleaned/q_merged_data_small_5.csv')

@st.cache_data
def load_q_k_key():
    return pd.read_csv('data/cleaned/key_quartier_kreis.csv')

# IMPUTE_STRATEGY = 'mean'
IMPUTE_STRATEGY = 'median'

@st.cache_data
def load_X_scaled():
    df = load_dataframe()
    high_card_cols = [col for col in df.select_dtypes(include=["object", "category"]).columns
                  if df[col].nunique() > 50]
    df_encoded = pd.get_dummies(df.drop(columns=["Quartier", "Year", "Kreis #"]).select_dtypes(include=["object", "category"]))
    X = pd.concat([df_encoded, df.drop(columns=["Quartier", "Year", "Kreis #"]).select_dtypes(include=[np.number])], axis=1)
    X_cols = X.columns
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = SimpleImputer(strategy=IMPUTE_STRATEGY).fit_transform(X_scaled)
    # X_cols = df.drop(columns=["Quartier", 'Year', 'Kreis #']).select_dtypes(include=[np.number]).columns
    return X_scaled, X_cols
