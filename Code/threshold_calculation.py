import pandas as pd
import numpy as np
from intermediate_results import *


def calculate_strength():
    dataframe = pd.read_csv('../Results/adjacency_matrix.csv', index_col=0)
    array = dataframe.to_numpy()
    col_sum_list = []
    row_sum_list = []

    for n, row in enumerate(array):
        col_sum = np.sum(array[:, n])
        col_sum_list.append(col_sum)
        row_sum = np.sum(row)
        row_sum_list.append(row_sum)

    strength_dict = dict(zip(Variables.all_stations_list, (col_sum_list[n] + row_sum_list[n] for n, row in enumerate(array))))
    print(strength_dict)

def calculate_degree():
    dataframe = pd.read_csv('../Results/adjacency_matrix.csv', index_col=0)
    array = dataframe.to_numpy()
    non_zero_array = array > 0

    col_sum_list = []
    row_sum_list = []

    for n, row in enumerate(non_zero_array):
        col_sum = np.sum(non_zero_array[:, n])
        col_sum_list.append(col_sum)
        row_sum = np.sum(row)
        row_sum_list.append(row_sum)

    degree_dict = dict(zip(Variables.all_stations_list, (col_sum_list[n] + row_sum_list[n] for n, row in enumerate(non_zero_array))))
    print(degree_dict)

def get_station_list_with_threshold():
    final_station_list = []
    for element in Variables.strength.items():
        if element[1] > 963:
            final_station_list.append(element[0])

    print(len(final_station_list)) # 275

    return final_station_list   # final_station_list

def apply_threshold_to_adjacency_matrix():
    dataframe = pd.read_csv('../Results/adjacency_matrix.csv', index_col=0)

    array = dataframe.to_numpy()
    for row in range(0, len(array)):
        for col in range(0, len(array)):
            if (col > row):
                array[col][row] += array[row][col]
                array[row][col] = 0
    df = pd.DataFrame(array, columns=Variables.all_stations_list, index=Variables.all_stations_list)

    df_less_columns = df[df.columns.intersection(Variables.final_station_list)]
    result = df_less_columns[(df_less_columns.index.isin(Variables.final_station_list))]

    return result.to_csv('adjacency_matrix_with_threshold.csv')

if __name__ == "__main__":
    calculate_strength()
    calculate_degree()
    get_station_list_with_threshold()
    apply_threshold_to_adjacency_matrix()
