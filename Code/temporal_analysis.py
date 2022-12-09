import pandas as pd
from datetime import *
from intermediate_results import *


def create_temporal_dataframe():
    df = pd.read_csv('../Results/202205-divvy-tripdata.csv')
    df = df[['started_at', 'ended_at', 'start_station_name', 'end_station_name']]
    df = df.dropna()

    new_df = pd.DataFrame(columns=['started_at', 'ended_at', 'start_station_name', 'end_station_name'])

    for count, row in enumerate(df.itertuples()):
        start_time = datetime.strptime(row.started_at, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(row.ended_at, "%Y-%m-%d %H:%M:%S")
        new_row = pd.Series([start_time.hour, end_time.hour, row.start_station_name, row.end_station_name], index=['started_at', 'ended_at', 'start_station_name', 'end_station_name'])
        new_df = new_df.append(new_row, ignore_index=True)

    return new_df.to_csv('temporal_dataset.csv')

def sort_temporal_data():
    df = pd.read_csv('../Results/temporal_dataset.csv')
    df = df.dropna()
    hours = [str(x) for x in list(range(24))]

    rent_df = pd.DataFrame(0, index=Variables.all_stations_list, columns=hours)
    return_df = pd.DataFrame(0, index=Variables.all_stations_list, columns=hours)

    for row in df.itertuples():
        start_time = row.started_at
        end_time = row.ended_at
        start_station_id = row.start_station_name
        end_station_id = row.end_station_name
        rent_df[str(start_time)][str(start_station_id)] += 1
        return_df[str(end_time)][str(end_station_id)] += 1

    rent_df = rent_df[(rent_df.index.isin(Variables.final_station_list))]
    return_df = return_df[(return_df.index.isin(Variables.final_station_list))]

    return rent_df.to_csv('rent_df.csv'), return_df.to_csv('return_df.csv')

def normalize_rents_and_returns():
    rent_df = pd.read_csv('../Results/rent_df.csv', index_col=0)
    return_df = pd.read_csv('../Results/return_df.csv', index_col=0)
    rent_df_t = rent_df.transpose()
    return_df_t = return_df.transpose()

    for column in rent_df_t.columns:
        rent_df_t[column] = rent_df_t[column] / rent_df_t[column].max()

    for column in return_df_t.columns:
        return_df_t[column] = return_df_t[column] / return_df_t[column].max()

    rent_df_t = rent_df_t.transpose()
    return_df_t = return_df_t.transpose()

    return rent_df_t.to_csv('rent_df_normalized.csv'), return_df_t.to_csv('return_df_normalized.csv')

if __name__ == "__main__":
    create_temporal_dataframe()
    sort_temporal_data()
    normalize_rents_and_returns()
