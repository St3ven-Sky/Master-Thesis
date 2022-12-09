import pandas as pd
from intermediate_results import *


def get_station_list():
    df = pd.read_csv('../Results/202205-divvy-tripdata.csv')
    df = df[['start_station_name', 'end_station_name']]
    df = df.dropna()
    all_stations_list = []

    for row in df.iterrows():
        start_station_id = row[1][0]
        end_station_id = row[1][1]
        all_stations_list.append(start_station_id)
        all_stations_list.append(end_station_id)

    return set(all_stations_list)

def sort_station_locations():
    df = pd.read_csv('../Results/Divvy_Bicycle_Stations.csv')
    df = df[['Station Name', 'Latitude', 'Longitude']]
    df.rename(columns={'Station Name': 'Station_Name'}, inplace=True)
    df.sort_values('Station_Name')
    station_location_dict = dict()
    for row in df.iterrows():
        if row[1][0] in Variables.final_station_list:
            station_location_dict[row[1][0]] = {'latitude': row[1][1], 'longitude': row[1][2]}

    station_location_df = pd.DataFrame.from_dict(station_location_dict, orient='index')

    return station_location_df.to_csv('station_locations.csv')

def create_adjacency_matrix():
    df = pd.read_csv('../Results/202205-divvy-tripdata.csv')
    df = df[['start_station_name', 'end_station_name']]
    df = df.dropna()

    adjacency_matrix = pd.DataFrame(0, index=Variables.all_stations_list,
                                    columns=Variables.all_stations_list)

    for row in df.iterrows():
        start_station_id = row[1][0]
        end_station_id = row[1][1]
        if start_station_id == end_station_id:
            adjacency_matrix[start_station_id][end_station_id] += 0
        else:
            adjacency_matrix[start_station_id][end_station_id] += 1

    return adjacency_matrix.to_csv('adjacency_matrix.csv')

if __name__ == "__main__":
    get_station_list()
    sort_station_locations()
    create_adjacency_matrix()