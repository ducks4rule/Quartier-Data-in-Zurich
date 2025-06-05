import pandas as pd
import numpy as np
import os
import functools

switch = False
if __name__ == "__main__":
    switch = True



NAMES_DFs = [
    'q_bevoelkerung_nationalitaeten',
    'q_leerflaechen',
    'q_motorisierungsgrad',
    'q_neuzulassung_pkw',
    'q_pkw_haushalt',
    'q_pkw_quartier',
    'q_sozialhilfe',
    'q_todesfaelle',
    'q_wohnungspreis',
    'q_wohnunsdichte',
    'q_personen_alter_nat_wohnen',
    'q_personen_alter_zivil',
    'q_einkommen_steuer',
    'q_religion',
    'q_geburten',
]

def save_to_csv(df, name, switch=False):
    if switch:
        output_path = f'data/cleaned/{name}.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving to: {os.path.abspath(output_path)}")  # Debug print
        df.to_csv(output_path, index=False)
        # clear memory
    del df

def weighted_mean(s, df, col: str):
    return (s * df.loc[s.index, col]).sum() / df.loc[s.index, col].sum()

def weighted_median(s, df, col: str):
    weights = df.loc[s.index, col]
    values = s.values
    sorted_idx = np.argsort(values)
    values, weights = values[sorted_idx], weights.values[sorted_idx]
    cum_weights = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    return values[cum_weights >= cutoff][0]


df_bev_u_nat = pd.read_csv('data/bev_u_nationalitaeten_p_quartier.csv') # done
df_leerflaechen = pd.read_csv('data/leerflaechen_p_quartier.csv') # done
df_motorisierungsgrad = pd.read_csv('data/motorisierungsgrad_p_quartier.csv') # done  TODO: what is the 'grad'?
df_neuzulassung_pkw = pd.read_csv('data/neuzulassung_pkw_p_quartier.csv') # done
df_pkw_haush = pd.read_csv('data/pkw_p_haushalt_p_quartier.csv') # done # TODO: what are the numbers?
df_pkw_quartier = pd.read_csv('data/pkw_p_quartier.csv') # done
df_sozialhilfe = pd.read_csv('data/sozialhilfe_p_quartier.csv') # done
df_todesfaelle = pd.read_csv('data/todesfaelle_p_quartier.csv') # done
df_wohnungspreis = pd.read_csv("data/wohnung_verkaufspreis_p_quartier.csv") # done
df_wohnunsdichte = pd.read_csv("data/wohnfläche_dichte_p_q.csv") # done
df_pers_1 = pd.read_csv("data/alter_nat_geb_p_q.csv") # done
df_pers_2_mod = pd.read_csv("data/cleaned/q_personen_alter_zivil.csv", usecols=['Year', 'Quartier', 'AlterVSort', 'AnzBestWir'])
df_einkommen_u_steuer = pd.read_csv("data/einkommen_steuer_p_q.csv") # done
df_geburten = pd.read_csv("data/geburtenr_p_q.csv") # done
df_religion = pd.read_csv("data/bev_religion.csv") # done # NOTE: only catholic and protestant and other. not more info

def inspect_data(df, switch=False, print_cols=True):
    if switch:
        if print_cols:
            for col in df.columns:
                print(f"{col} : \t{df[col].dtype}\t memory: {df[col].memory_usage(deep=True) * 1e-6:.4f} MB")
        print(f"Total memory usage: {df.memory_usage(deep=True).sum() * 1e-6:.4f} MB")

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

        for col in df.select_dtypes(include=[object]).columns:
            if df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit() or str(x)== '.').all():
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



common_rename = {'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'AnzBestWirk': 'AnzBestWir'}
common_drop = ['QuarCd', 'QuarSort']


# df_bev_u_nat cleaning
df_bev_u_nat = df_bev_u_nat.rename(columns=common_rename)
df_bev_u_nat = df_bev_u_nat.rename(columns={'AnzNat': 'Num Nationalities', 'AnzBestWir': 'Inhabitants p Q'})
df_bev_u_nat['Inhabitants total'] = df_bev_u_nat.groupby('Year')['Inhabitants p Q'].transform('sum')
df_bev_u_nat = df_bev_u_nat.drop(columns=['QuarSort', 'QuarCd', 'DatenstandCd'])

df = df_bev_u_nat
df = clean_data(df, switch=switch)
df_bev_u_nat = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[0], switch=switch)



# df_leerflaechen cleaning
df_leerflaechen = df_leerflaechen.rename(columns=common_rename)
df_leerflaechen = df_leerflaechen.rename(columns={'KreisSort': 'Kreis #', 'Nutzung': 'Usage', 'Leerflaeche': 'Free Space', 'BueroNettofl': 'Office Space', 'BueroLeerProz': 'Office Space %'})
df_leerflaechen = df_leerflaechen.drop(columns=['QuarSort', 'KreisLang'])
df = df_leerflaechen
df = clean_data(df, switch=switch)
df_leerflaechen = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[1], switch=switch)


# df_motorisierungsgrad cleaning
df_motorisierungsgrad = df_motorisierungsgrad.rename(columns=common_rename)
df_motorisierungsgrad = df_motorisierungsgrad.rename(columns={'Motorisierungsgrad': 'Motorization Rate'})
df_motorisierungsgrad = df_motorisierungsgrad.drop(columns=['QuarSort', 'QuarCd', 'StichtagDat'])
df = df_motorisierungsgrad
df = clean_data(df, switch=switch)
df_motorisierungsgrad = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[2], switch=switch)

    
# df_neuzulassung_pkw cleaning
df_neuzulassung_pkw = df_neuzulassung_pkw.rename(columns=common_rename)
df_neuzulassung_pkw = df_neuzulassung_pkw.rename(columns={'FhRechtsformLang': 'Legal Form New PKW', 'FzTreibstoffAgg_noDM': 'Fuel Type New PKW', 'FzAnz': 'Num New PKW / Q'})
df_neuzulassung_pkw = df_neuzulassung_pkw.drop(columns=['QuarSort', 'QuarCd', 'StichtagDat', 'KreisLang', 'KreisCd', 'FzTreibstoffAggCd_noDM', 'KreisSort', 'FhRechtsformCd'])
df_neuzulassung_pkw['Num New PKW total'] = df_neuzulassung_pkw.groupby(['Year'])['Num New PKW / Q'].transform('sum')
df = df_neuzulassung_pkw
df = clean_data(df, switch=switch)
df_neuzulassung_pkw = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[3], switch=switch)


# df_pkw_haush cleaning
df_pkw_haush = df_pkw_haush.rename(columns=common_rename)
df_pkw_haush = df_pkw_haush.rename(columns={'AnzPWHHSort_noDM': 'Num PKW / HH', 'AnzHH': 'Num Households'})
df_pkw_haush = df_pkw_haush.drop(columns=['QuarSort', 'QuarCd', 'StichtagDat', 'KreisLang', 'KreisCd', 'KreisSort', 'AnzPWHHCd_noDM', 'AnzPWHHLang_noDM'])
df = df_pkw_haush
df = clean_data(df, switch=switch)
df_pkw_haush = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[4], switch=switch)


# df_pkw_quartier cleaning
df_pkw_quartier = df_pkw_quartier.rename(columns=common_rename)
df_pkw_quartier = df_pkw_quartier.rename(columns={'FzTreibstoffAgg_noDM': 'Fuel Type', 'FhRechtsformLang': 'Legal Form', 'FzAnz': 'Num PKW / Q'})
df_pkw_quartier = df_pkw_quartier.drop(columns=['QuarSort', 'QuarCd', 'StichtagDat', 'KreisLang', 'KreisCd', 'FhRechtsformCd','FzTreibstoffAggCd_noDM', 'KreisSort'])
df_pkw_quartier['Num PKW total'] = df_pkw_quartier.groupby(['Year'])['Num PKW / Q'].transform('sum')
df = df_pkw_quartier
df = clean_data(df, switch=switch)
df_pkw_quartier = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[5], switch=switch)


# df_sozialhilfe cleaning
df_sozialhilfe = df_sozialhilfe.rename(columns={'Jahr': 'Year', 'Raum': 'Quartier', 'SH_Quote': 'Social Aid Rate', 'SH_Beziehende': 'Num Social Aid Recipients'})
df_sozialhilfe = df_sozialhilfe.drop(columns=['ID_Quartier', 'Zivil_Bevölkerung'])
df = df_sozialhilfe
df = clean_data(df, switch=switch)
df_sozialhilfe = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[6], switch=switch)


# df_todesfaelle cleaning
df_todesfaelle = df_todesfaelle.rename(columns=common_rename)
df_todesfaelle = df_todesfaelle.rename(columns={'AnzSterWir': 'Num Deaths p Q'})
df_todesfaelle = df_todesfaelle.drop(columns=['QuarSort'])
df_todesfaelle['Num Deaths total'] = df_todesfaelle.groupby(['Year'])['Num Deaths p Q'].transform('sum')
df_todesfaelle['Deaths Local in %'] = df_todesfaelle['Num Deaths p Q'] / df_todesfaelle['Num Deaths total']
df = df_todesfaelle
df = clean_data(df, switch=switch)
df_todesfaelle = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[7], switch=switch)

# df_wohnungspreis cleaning
df_wohnungspreis = df_wohnungspreis.rename(columns={'Stichtagdatjahr': 'Year', 'RaumLang': 'Quartier', 'AnzZimmerLevel2Cd_noDM': 'Num Rooms', 'HAPreisWohnflaeche': 'Price Living Space p Q', 'HAMedianPreis': 'Median Price p Q', 'HASumPreis': 'Volume Apartement Prices p Q'})
df_wohnungspreis = df_wohnungspreis.drop(columns=['RaumCd', 'RaumSort', 'AnzZimmerLevel2Sort_noDM', 'AnzZimmerLevel2Lang_noDM', 'AnzHA'])
df_wohnungspreis['Price Living Space total'] = df_wohnungspreis.groupby(['Year'])['Price Living Space p Q'].transform('sum')
df_wohnungspreis['Median Price total'] = df_wohnungspreis.groupby(['Year'])['Median Price p Q'].transform('sum')
# df_wohnungspreis.loc[df_wohnungspreis['Sum Price p Q'] == 'K', 'Sum Price p Q'] = 0
df = df_wohnungspreis
df = clean_data(df, switch=switch)
df_wohnungspreis = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[8], switch=switch)

# df_wohnunsdichte cleaning
df_wohnunsdichte = df_wohnunsdichte.rename(columns=common_rename)
df_wohnunsdichte = df_wohnunsdichte.drop(columns=['QuarSort', 'QuarCd', 'KreisLang', 'KreisCd', 'KreisSort', 'EigentuemerSSZPubl1Sort', 'EigentuemerSSZPubl1Cd', 'AnzBestWir'])
df_wohnunsdichte = df_wohnunsdichte.rename(columns={'EigentuemerSSZPubl1Lang': 'Owner Type Living', 'AnzWhgStat': 'Num Apartments p Q', 'Wohnflaeche': 'Living Space Local', 'PersProWhg_noDM': 'Num Persons p Ap p Q', 'WohnungsflProPers_noDM': 'Living Space p Person p Q'})
df_wohnunsdichte['Num Apartments total'] = df_wohnunsdichte.groupby(['Year'])['Num Apartments p Q'].transform('sum')
df_wohnunsdichte['Living Space Local total'] = df_wohnunsdichte.groupby(['Year'])['Living Space Local'].transform('sum')
df_wohnunsdichte['Living Space Local in %'] = df_wohnunsdichte['Living Space Local'] / df_wohnunsdichte['Living Space Local total']
df_wohnunsdichte['Num Apartments in %'] = df_wohnunsdichte['Num Apartments p Q'] / df_wohnunsdichte['Num Apartments total']
df_wohnunsdichte['Living Space p Person total'] = df_wohnunsdichte.groupby(['Year'])['Living Space p Person p Q'].transform('sum')
df_wohnunsdichte['Living Space p Person in %'] = df_wohnunsdichte['Living Space p Person p Q'] / df_wohnunsdichte['Living Space p Person total']
df = df_wohnunsdichte
df = clean_data(df, switch=switch)
df_wohnunsdichte = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[9], switch=switch)

# df_pers_1 cleaning
df_pers_1 = df_pers_1.rename(columns=common_rename)
df_pers_1 = df_pers_1.rename(columns={'AlterV10Cd': 'Age Group (10)', 'GebCHAUCd': 'Birth CH or not', 'HerkunftCd': 'Origin CH or not', 'EigentuemerSSZPubl1Lang': 'Owner Type Living', 'AnzBestWir': 'Num p Age Group p Q'})
df_pers_1 = df_pers_1.drop(columns=['QuarSort', 'QuarCd', 'KreisLang', 'KreisCd', 'KreisSort', 'AlterV10Sort', 'AlterV10Kurz', 'AlterV10Lang', 'GebCHAUSort', 'GebCHAULang', 'HerkunftSort', 'HerkunftLang', 'EigentuemerSSZPubl1Cd', 'EigentuemerSSZPubl1Sort'])
df_pers_1['Num p Age Group total'] = df_pers_1.groupby(['Year'])['Num p Age Group p Q'].transform('sum')
df_pers_1['Mean Age Group p Q'] = (
    df_pers_1.groupby(['Year', 'Quartier'])['Age Group (10)']
    .transform(lambda x: weighted_mean(x, df_pers_1, 'Num p Age Group p Q'))
)
df_pers_1['Mean Age Group total'] = (
    df_pers_1.groupby(['Year'])['Age Group (10)']
    .transform(lambda x: weighted_mean(x, df_pers_1, 'Num p Age Group total'))
)
df_pers_1['Median Age Group p Q'] = (
    df_pers_1.groupby(['Year', 'Quartier'])['Age Group (10)']
    .transform(lambda x: weighted_median(x, df_pers_1, 'Num p Age Group p Q'))
)
df_pers_1['Median Age Group total'] = (
    df_pers_1.groupby(['Year'])['Age Group (10)']
    .transform(lambda x: weighted_median(x, df_pers_1, 'Num p Age Group total'))
)

df = df_pers_1
df = clean_data(df, switch=switch)
df_pers_1 = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[10], switch=switch)

df_pers_1_mod = df_pers_1.copy()
df_pers_1_mod = df_pers_1_mod.drop(columns=['Age Group (10)', 'Birth CH or not', 'Origin CH or not', 'Owner Type Living', 'Num p Age Group p Q', 'Num p Age Group total'])
df = df_pers_1_mod
df = clean_data(df, switch=switch)
df_pers_1_mod = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[10] + '_small', switch=switch)

df_pers_2_mod['Mean Age p Q'] = (
    df_pers_2_mod.groupby(['Year', 'Quartier'])['AlterVSort']
    .transform(lambda x: weighted_mean(x, df_pers_2_mod, 'AnzBestWir'))
)
df_pers_2_mod['Mean Age total'] = (
    df_pers_2_mod.groupby(['Year'])['AlterVSort']
    .transform(lambda x: weighted_mean(x, df_pers_2_mod, 'AnzBestWir'))
)
df_pers_2_mod['Median Age p Q'] = (
    df_pers_2_mod.groupby(['Year', 'Quartier'])['AlterVSort']
    .transform(lambda x: weighted_median(x, df_pers_2_mod, 'AnzBestWir'))
)
df_pers_2_mod['Median Age total'] = (
    df_pers_2_mod.groupby(['Year'])['AlterVSort']
    .transform(lambda x: weighted_median(x, df_pers_2_mod, 'AnzBestWir'))
)
df_pers_2_mod = df_pers_2_mod.drop(columns=['AlterVSort', 'AnzBestWir'])
df = df_pers_2_mod
df = clean_data(df, switch=switch)
df_pers_2_mod = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[11] + '_small', switch=switch)

# df_einkommen_u_steuer cleaning
df_einkommen_u_steuer = df_einkommen_u_steuer.rename(columns=common_rename)
df_einkommen_u_steuer = df_einkommen_u_steuer.rename(columns={'SteuerTarifLang': 'Tax Rate', 'SteuerEinkommen_p50': 'Income Median p Q', 'SteuerEinkommen_p25': 'Income 25% p Q', 'SteuerEinkommen_p75': 'Income 75% p Q'})
df_einkommen_u_steuer = df_einkommen_u_steuer.drop(columns=['QuarSort', 'QuarCd', 'SteuerTarifSort', 'SteuerTarifCd'])
df_einkommen_u_steuer['Income Median total'] = df_einkommen_u_steuer.groupby(['Year'])['Income Median p Q'].transform('sum')
df_einkommen_u_steuer['Income 25% total'] = df_einkommen_u_steuer.groupby(['Year'])['Income 25% p Q'].transform('sum')
df_einkommen_u_steuer['Income 75% total'] = df_einkommen_u_steuer.groupby(['Year'])['Income 75% p Q'].transform('sum')


df = df_einkommen_u_steuer
df = clean_data(df, switch=switch)
df_einkommen_u_steuer = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[12], switch=switch)


# df_geburten cleaning
df_geburten = df_geburten.rename(columns={'EreignisDatJahr': 'Year', 'QuarLang': 'Quartier', 'AlterVMutterCd': 'Age of Mother', 'SexCd': 'Sex Child', 'AnzGebuWir': 'Num Births p Q'})
df_geburten = df_geburten.drop(columns=['QuarCd'])
df_geburten = df_geburten.drop(columns=['HerkunftCd', 'HerkunftMutterCd'])
df_geburten['Num Births total'] = df_geburten.groupby(['Year'])['Num Births p Q'].transform('sum')
df_geburten['Mean Age of Mother p Q'] = (
    df_geburten.groupby(['Year', 'Quartier'])['Age of Mother']
    .transform(lambda x: weighted_mean(x, df_geburten, 'Num Births p Q'))
)
df_geburten['Mean Age of Mother total'] = (
    df_geburten.groupby(['Year'])['Age of Mother']
    .transform(lambda x: weighted_mean(x, df_geburten, 'Num Births total'))
)
df_geburten['Median Age of Mother p Q'] = (
    df_geburten.groupby(['Year', 'Quartier'])['Age of Mother']
    .transform(lambda x: weighted_median(x, df_geburten, 'Num Births p Q'))
)
df_geburten['Median Age of Mother total'] = (
    df_geburten.groupby(['Year'])['Age of Mother']
    .transform(lambda x: weighted_median(x, df_geburten, 'Num Births total'))
)
df_geburten.drop(columns=['Age of Mother', 'Sex Child', 'Num Births p Q' ], inplace=True)

df = df_geburten
df = clean_data(df, switch=switch)
df_geburten = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[14], switch=switch)


# df_religion cleaning
df_religion = df_religion.rename(columns=common_rename)
df_religion = df_religion.rename(columns={'HerkunftLang': 'Origin', 'Kon2AggLang_noDM': 'Religion', 'AnzBestWir': 'Num p Religion'})
df_religion = df_religion.drop(columns=['QuarSort', 'StatZoneSort', 'StatZoneLang', 'KreisSort', 'HerkunftSort', 'HerkunftCd', 'Kon2AggCd_noDM', 'Kon2AggSort_noDM'])

df = df_religion
df = clean_data(df, switch=switch)
df_religion = df
# inspect_data(df, switch=switch)
save_to_csv(df, NAMES_DFs[13], switch=switch)



all_dfs = [df_bev_u_nat, df_leerflaechen, #df_motorisierungsgrad, #df_neuzulassung_pkw, df_pkw_haush,
           df_pkw_quartier,
           df_sozialhilfe, df_todesfaelle,
           df_wohnungspreis,
           # df_wohnunsdichte,
           # df_pers_1_mod,
           df_pers_2_mod,
           df_einkommen_u_steuer,
           df_geburten] #, df_religion]

# all_dfs = [df_leerflaechen, df_wohnungspreis, df_wohnunsdichte]

# for df in all_dfs:
#     inspect_data(df, switch=switch, print_cols=False)
    

# Merge all dataframes
df_merged = functools.reduce(lambda left, right: pd.merge(left, right, on=['Year', 'Quartier'], how='outer'), all_dfs)
inspect_data(df_merged, switch=switch, print_cols=False)
# inspect_data(df_merged, switch=switch, print_cols=True)
df_merged = clean_data(df_merged, switch=switch)
inspect_data(df_merged, switch=switch, print_cols=True)
# inspect_data(df_merged, switch=switch, print_cols=False)
save_to_csv(df_merged, 'q_merged_data_small_5', switch=switch)
