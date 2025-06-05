import pandas as pd
import numpy as np
import streamlit as st
import os
import functools
import gc
import psutil

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



df_bev_u_nat = pd.read_csv('data/bev_u_nationalitaeten_p_quartier.csv') # done
df_leerflaechen = pd.read_csv('data/leerflaechen_p_quartier.csv') # done
df_motorisierungsgrad = pd.read_csv('data/motorisierungsgrad_p_quartier.csv') # done  TODO: what is the 'grad'?
df_neuzulassung_pkw = pd.read_csv('data/neuzulassung_pkw_p_quartier.csv') # done
df_pkw_haush = pd.read_csv('data/pkw_p_haushalt_p_quartier.csv') # done # TODO: what are the numbers?
df_pkw_quartier = pd.read_csv('data/pkw_p_quartier.csv') # done
df_sozialhilfe = pd.read_csv('data/sozialhilfe_p_quartier.csv') # done
df_todesfaelle = pd.read_csv('data/todesfaelle_p_quartier.csv') # done
# df_arbeitstaetten = pd.read_csv("data/arbeitsstaetten_beschaeftigte_jahr_quartier_noga.csv")
df_wohnungspreis = pd.read_csv("data/wohnung_verkaufspreis_p_quartier.csv") # done
df_wohnunsdichte = pd.read_csv("data/wohnfläche_dichte_p_q.csv") # done
df_pers_1 = pd.read_csv("data/alter_nat_geb_p_q.csv") # done
df_pers_2 = pd.read_csv("data/bev_zivil_sex_alter_p_q.csv") # done
df_einkommen_u_steuer = pd.read_csv("data/einkommen_steuer_p_q.csv") # done
df_geburten = pd.read_csv("data/geburtenr_p_q.csv") # done
df_religion = pd.read_csv("data/bev_religion.csv") # done # NOTE: only catholic and protestant and other. not more info

# dfs to be ignored for now
df_bev = pd.read_csv('data/bevoelkerung_p_quartier.csv') # NOTE: starts earlier than everything else
df_nat = pd.read_csv('data/nationalitaet_p_quartier.csv') # NOTE: country, reagion, continent
df_pkw_marke = pd.read_csv('data/pkw_marke_p_quartier.csv') # NOTE: brand and number of cars -- maybe not useful
df_todesfaelle_2 = pd.read_csv("data/todesfaelle_alter_sex_herk.csv") # NOTE: too much details probably
df_migration = pd.read_csv("data/migrations_status_p_q.csv") # NOTE: a lot of details, need to think about utility / no year data





# names of quartiere
quarlangs = df_bev_u_nat['QuarLang'].unique()
df_key_quartier_kreis = df_leerflaechen[['QuarLang', 'KreisLang']].drop_duplicates()
df_key_quartier_kreis = df_key_quartier_kreis.rename(columns={'KreisLang': 'Kreis', 'QuarLang': 'Quartier'})

if __name__ == "__main__":
    df_key_quartier_kreis.to_csv('data/cleaned/key_quartier_kreis.csv', index=False)



# df_bev_u_nat -- renaming and dropping columns
df_bev_u_nat = df_bev_u_nat.drop(columns=['QuarSort', 'QuarCd', 'DatenstandCd'])
df_bev_u_nat['Total Inhabitants'] = df_bev_u_nat.groupby(['StichtagDatJahr'])['AnzBestWir'].transform('sum')
df_bev_u_nat = df_bev_u_nat.rename(columns={'StichtagDatJahr': 'Year', 'AnzBestWir': 'Inhabitants', 'AnzNat': 'Num Nationalities', 'QuarLang': 'Quartier'})

save_to_csv(df_bev_u_nat, NAMES_DFs[0], switch=switch)

# adding total and percentage columns to df_todesfaelle and renaming, dropping 
tod_total = df_todesfaelle.groupby('StichtagDatJahr', as_index=False)['AnzSterWir'].transform('sum')
df_todesfaelle['Total Deaths'] = tod_total
df_todesfaelle['Deaths Percent'] = df_todesfaelle['AnzSterWir'] / df_todesfaelle['Total Deaths']
df_todesfaelle = df_todesfaelle.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'AnzSterWir': 'Deaths Local'})
df_todesfaelle = df_todesfaelle.drop(columns=['QuarSort'])

save_to_csv(df_todesfaelle, NAMES_DFs[7], switch=switch)

# df_geburten -- new features, renaming and dropping columns
def weighted_mean(s):
    return (s * df_geburten.loc[s.index, 'AnzGebuWir']).sum() / df_geburten.loc[s.index, 'AnzGebuWir'].sum()
df_geburten['Mean Age Mother'] = (
    df_geburten.groupby(['EreignisDatJahr', 'QuarLang'])['AlterVMutterCd']
        .transform(weighted_mean)
)
def weighted_median(s):
    weights = df_geburten.loc[s.index, 'AnzGebuWir']
    values = s.values
    sorted_idx = np.argsort(values)
    values, weights = values[sorted_idx], weights.values[sorted_idx]
    cum_weights = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    return values[cum_weights >= cutoff][0]

df_geburten['Median Age Mother'] = (
    df_geburten.groupby(['EreignisDatJahr', 'QuarLang'])['AlterVMutterCd']
        .transform(weighted_median)
)
df_geburten['Total Births'] = df_geburten.groupby('EreignisDatJahr')['AnzGebuWir'].transform('sum')
df_geburten['Births Percent'] = df_geburten['AnzGebuWir'] / df_geburten['Total Births']
df_geburten = df_geburten.rename(columns={'EreignisDatJahr': 'Year', 'QuarLang': 'Quartier', 'AnzGebuWir': 'Births Local', 'AlterVMutterCd': 'Age Mother', 'SexCd': 'Sex Child', 'LebensfaehigkeitCd': 'Child Alife'})
df_geburten['Child Alife'] = df_geburten['Child Alife'].replace({'J': 1, 'N': 0})
df_geburten = df_geburten.drop(columns=['QuarCd', 'HerkunftCd', 'HerkunftMutterCd'])

save_to_csv(df_geburten, NAMES_DFs[14], switch=switch)

# df_leerflaechen -- renaming and dropping columns
df_leerflaechen['Total Local Vacancy'] = df_leerflaechen.groupby(['StichtagDatJahr', 'QuarLang'])['Leerflaeche'].transform('sum')
df_leerflaechen['Vacancy Percent'] = df_leerflaechen['Leerflaeche'] / df_leerflaechen['Total Local Vacancy']
df_leerflaechen = df_leerflaechen.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'Leerflaeche': 'Local Vacancy', 'Nutzung': 'Usage Vacancy', 'BueroNettofl': 'Net Office Space', 'BueroLeerProz': 'Office Vacancy Percent'})
df_leerflaechen = df_leerflaechen.drop(columns=['QuarSort', 'KreisSort', 'KreisLang'])

save_to_csv(df_leerflaechen, NAMES_DFs[1], switch=switch)

# df_motorisierungsgrad -- renaming and dropping columns
df_motorisierungsgrad = df_motorisierungsgrad.drop(columns=['StichtagDat', 'QuarCd', 'QuarSort'])
df_motorisierungsgrad['Total Motorization'] = df_motorisierungsgrad.groupby(['StichtagDatJahr'])['Motorisierungsgrad'].transform('sum')
df_motorisierungsgrad['Motorization Percent'] = df_motorisierungsgrad['Motorisierungsgrad'] / df_motorisierungsgrad['Total Motorization']
df_motorisierungsgrad = df_motorisierungsgrad.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'Motorisierungsgrad': 'Local Motorization'})

save_to_csv(df_motorisierungsgrad, NAMES_DFs[2], switch=switch)

# df_neuzulassung_pkw -- renaming and dropping columns
df_neuzulassung_pkw = df_neuzulassung_pkw.drop(columns=['StichtagDat', 'QuarCd', 'QuarSort', 'KreisLang', 'KreisCd', 'KreisSort', 'FhRechtsformCd','FzTreibstoffAggCd_noDM'])
df_neuzulassung_pkw['Local New Registrations Fuel'] = df_neuzulassung_pkw.groupby(['StichtagDatJahr', 'QuarLang','FzTreibstoffAgg_noDM'])['FzAnz'].transform('sum')
df_neuzulassung_pkw['Local New Registrations'] = df_neuzulassung_pkw.groupby(['StichtagDatJahr', 'QuarLang'])['FzAnz'].transform('sum')
df_neuzulassung_pkw['Total Registrations'] = df_neuzulassung_pkw.groupby(['StichtagDatJahr'])['FzAnz'].transform('sum')
df_neuzulassung_pkw['New Registrations Percent'] = df_neuzulassung_pkw['FzAnz'] / df_neuzulassung_pkw['Total Registrations']
df_neuzulassung_pkw = df_neuzulassung_pkw.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'FzAnz': 'New Registrations', 'FhRechtsformLang': 'Legal Form', 'FzTreibstoffAgg_noDM': 'Fuel Type'})

save_to_csv(df_neuzulassung_pkw, NAMES_DFs[3], switch=switch)

# df_pkw_quartier -- renaming and dropping columns -- same as df_neuzulassung_pkw
df_pkw_quartier = df_pkw_quartier.drop(columns=['StichtagDat', 'QuarCd', 'QuarSort', 'KreisLang', 'KreisCd', 'KreisSort', 'FhRechtsformCd','FzTreibstoffAggCd_noDM'])
df_pkw_quartier['Local Cars'] = df_pkw_quartier.groupby(['StichtagDatJahr', 'QuarLang'])['FzAnz'].transform('sum')
df_pkw_quartier['Total Cars'] = df_pkw_quartier.groupby(['StichtagDatJahr'])['FzAnz'].transform('sum')
df_pkw_quartier['Cars Percent'] = df_pkw_quartier['FzAnz'] / df_pkw_quartier['Total Cars']
df_pkw_quartier = df_pkw_quartier.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'FzAnz': 'Cars', 'FhRechtsformLang': 'Legal Form', 'FzTreibstoffAgg_noDM': 'Fuel Type'})

save_to_csv(df_pkw_quartier, NAMES_DFs[5], switch=switch)

# df_pkw_haush -- renaming and dropping columns
df_pkw_haush = df_pkw_haush.drop(columns=['StichtagDat', 'QuarCd', 'QuarSort', 'KreisLang', 'KreisCd', 'KreisSort', 'AnzPWHHSort_noDM'])
df_pkw_haush = df_pkw_haush.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'AnzPWHHCd_noDM': 'Cars per Household', 'AnzPWHHLang_noDM': 'Cars per Household Language', 'AnzPWHH': 'Local Cars per Household'})

save_to_csv(df_pkw_haush, NAMES_DFs[4], switch=switch)

# df_sozialhilfe -- renaming and dropping columns
df_sozialhilfe = df_sozialhilfe.drop(columns=['ID_Quartier'])
df_sozialhilfe = df_sozialhilfe.rename(columns={'Jahr': 'Year', 'Raum': 'Quartier', 'SH_Quote': 'Social Assistance Quota', 'SH_Beziehende': 'Social Assistance Recipients', 'Zivil_Bevölkerung': 'Civil Population'}) # TODO: check 'Civil Population' == 'Inhabitants'?

save_to_csv(df_sozialhilfe, NAMES_DFs[6], switch=switch)

# df_wohnungspreis -- renaming and dropping columns
df_wohnungspreis = df_wohnungspreis.drop(columns=['DatenstandCd', 'HAArtLevel1Sort', 'HAArtLevel1Cd', 'HAArtLevel1Lang', 'HASTWESort', 'HASTWECd', 'HASTWELang', 'RaumSort', "RaumCd", "AnzHA", "AnzZimmerLevel2Lang_noDM", "AnzZimmerLevel2Cd_noDM"])
df_wohnungspreis = df_wohnungspreis.rename(columns={'Stichtagdatjahr': 'Year', 'RaumLang': 'Quartier', 'AnzZimmerLevel2Sort_noDM': 'Number of Rooms', 'HAPreisWohnflaeche': 'Price per m²', 'HAMedianPreis': 'Median Price', 'HASumPreis': "Sum of prices (flats)"})

save_to_csv(df_wohnungspreis, NAMES_DFs[8], switch=switch)

# df_wohnunsdichte -- renaming and dropping columns
df_wohnunsdichte = df_wohnunsdichte.drop(columns=['DatenstandCd', 'QuarSort', 'QuarCd', 'KreisSort', 'KreisCd', 'KreisLang', 'EigentuemerSSZPubl1Cd', 'WohnungBewohntSort_noDM', 'WohnungBewohntCd_noDM', 'WohnungBewohntLang_noDM'])
df_wohnunsdichte = df_wohnunsdichte.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'EigentuemerSSZPubl1Sort': 'Owner Type, num', 'EigentuemerSSZPubl1Lang': 'Owner Type, name', 'AnzWhgStat': 'Number of Apartments', 'Wohnflaeche': 'Living Space', 'AnzBestWir': 'Inhabitants (appartments)', 'PersProWhg_noDM': 'Persons per Apartment', 'WohnungsflProPers_noDM': 'Living Space per Person'})

save_to_csv(df_wohnunsdichte, NAMES_DFs[9], switch=switch)

# df_pers_1 -- age, nationality, renaming and dropping columns
df_pers_1 = df_pers_1.drop(columns=['DatenstandCd', 'QuarSort', 'QuarCd', 'KreisSort', 'KreisCd', 'KreisLang', 'AlterV10Sort', 'AlterV10Kurz', 'AlterV10Lang', 'GebCHAUSort', 'GebCHAUCd', 'HerkunftSort', 'HerkunftCd', 'EigentuemerSSZPubl1Cd', 'EigentuemerSSZPubl1Sort'])
# df_pers_1['Average Local Age'] = df_pers_1.groupby(['StichtagDatJahr', 'QuarLang'])['AlterV10Cd'].transform('mean')
# df_pers_1['Inhabitants Age'] = df_pers_1.groupby(['StichtagDatJahr'])['AnzBestWir'].transform('sum')
# df_pers_1['Average Age'] = df_pers_1.groupby(['StichtagDatJahr'])['Inhabitants Age'].transform('mean')
df_pers_1 = df_pers_1.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'AnzBestWir': 'Inhabitants Local Age', 'AlterV10Cd': 'Age Group + 10', 'GebCHAULang': 'Country of Birth', 'HerkunftLang': 'Nationality', 'EigentuemerSSZPubl1Lang': 'Living Space Owner Type'})

save_to_csv(df_pers_1, NAMES_DFs[10], switch=switch)

df_pers_2 = df_pers_2.drop(columns=['AlterVCd', 'AlterV05Sort', 'AlterV05Cd', 'AlterV05Kurz', 'AlterV10Sort', 'AlterV10Kurz', 'AlterV20Sort', 'AlterV20Cd', 'AlterV20Kurz', 'SexCd', 'KreisCd' , 'KreisLang', 'QuarSort', 'QuarCd', 'Ziv2Sort', 'Ziv2Cd'])
# df_pers_2['AnzBestWir'].fillna(0, inplace=True)
# weighted_avg_age = (
#     df_pers_2
#     .groupby(['StichtagDatJahr','QuarLang', 'AlterVSort'])
#     .apply(lambda g: (g['AlterVSort'] * g['AnzBestWir']).sum() / g['AnzBestWir'].sum() if g['AnzBestWir'].sum() > 0 else 0)
#     .reset_index(name='Average Local Age')
# )
# df_pers_2 = df_pers_2.merge(weighted_avg_age, on=['QuarLang', 'StichtagDatJahr'], how='left')
df_pers_2['Total Inhabitants p Age'] = df_pers_2.groupby(['StichtagDatJahr', 'AlterVSort'])['AnzBestWir'].transform('sum')
# weighted_avg_age = (
#     df_pers_2
#     .groupby(['StichtagDatJahr', 'AlterVSort'])
#     .apply(lambda g: (g['AlterVSort'] * g['Total Inhabitants p Age']).sum() / g['Total Inhabitants p Age'].sum())
#     .reset_index(name='Average Age')
# )
# df_pers_2 = df_pers_2.merge(weighted_avg_age, on=['StichtagDatJahr'], how='left')
df_pers_2 = df_pers_2.rename(columns={'StichtagDatJahr': 'Year', 'AlterVSort': 'Age', 'AlterV10Cd': 'Age Group + 10', 'SexLang': 'Sex', 'QuarLang': 'Quartier', 'Ziv2Lang': 'Civil Status', 'AnzBestWir': 'Inhabitants Local'})

save_to_csv(df_pers_2, NAMES_DFs[11], switch=switch)

# df_einkommen_u_steuer -- renaming and dropping columns
df_einkommen_u_steuer = df_einkommen_u_steuer.drop(columns=['QuarSort', 'QuarCd', 'SteuerTarifSort', 'SteuerTarifCd'])
df_einkommen_u_steuer = df_einkommen_u_steuer.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'SteuerTarifLang': 'Tax Rate Type', 'SteuerEinkommen_p50': 'Median Taxable Income', 'SteuerEinkommen_p25': '25th Percentile Taxable Income', 'SteuerEinkommen_p75': '75th Percentile Taxable Income'})

save_to_csv(df_einkommen_u_steuer, NAMES_DFs[12], switch=switch)

# df_religion -- renaming and dropping columns
df_religion = df_religion.drop(columns=['StatZoneSort', 'StatZoneLang', 'QuarSort', 'KreisSort', 'HerkunftCd', 'HerkunftSort', 'Kon2AggSort_noDM', 'Kon2AggCd_noDM'])
df_religion = df_religion.rename(columns={'StichtagDatJahr': 'Year', 'QuarLang': 'Quartier', 'Kon2AggLang_noDM': 'Religion', 'AnzBestWir': 'Num Religion'})

save_to_csv(df_religion, NAMES_DFs[13], switch=switch)


# all data frames
# all_dfs = [
#     df_bev_u_nat,
#     df_leerflaechen,
#     df_motorisierungsgrad,
#     df_neuzulassung_pkw,
#     df_pkw_haush,
#     df_pkw_quartier,
#     df_sozialhilfe,
#     df_todesfaelle,
#     df_wohnungspreis,
#     df_wohnunsdichte,
#     df_pers_1,
#     df_pres_2,
#     df_einkommen_u_steuer,
#     df_religion,
#     df_geburten,
# ]






