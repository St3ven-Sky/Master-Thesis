import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from scipy import stats
import functools as ft
from cdlib import algorithms
import scipy.cluster.hierarchy as sch
from intermediate_results import *


def hierarchical_clustering():
    rents = pd.read_csv('../Results/rent_df_normalized.csv', index_col=0)
    rents = rents.dropna()
    rents_arr = rents.to_numpy()
    print(rents_arr)

    returns = pd.read_csv('../Results/return_df_normalized.csv', index_col=0)
    returns = returns.dropna()
    returns_arr = returns.to_numpy()
    print(returns_arr)

    sch.dendrogram(sch.linkage(rents_arr, method='ward'))
    hc_rents_normalised = AgglomerativeClustering(n_clusters=4)
    y_hc_rents_normalised = hc_rents_normalised.fit_predict(rents_arr)
    print(Counter(y_hc_rents_normalised))   # Counter({1: 93, 0: 72, 2: 59, 3: 45})
    plt.axhline(y=4.8)
    plt.show()
    print('y_hc_rents_normalised', y_hc_rents_normalised)

    sch.dendrogram(sch.linkage(returns_arr, method='ward'))
    hc_returns_normalised = AgglomerativeClustering(n_clusters=4)
    y_hc_returns_normalised = hc_returns_normalised.fit_predict(returns_arr)
    print(Counter(y_hc_returns_normalised))     # Counter({0: 105, 1: 102, 3: 40, 2: 25})
    plt.axhline(y=4.8)
    plt.show()
    print('y_hc_returns_normalised', y_hc_returns_normalised)

def visualize_rents():
    rents = pd.read_csv('../Results/rent_df_normalized.csv', index_col=0)
    errors = ['Springfield Ave & 63rd St', 'Michigan Ave & 113th St', 'Colfax Ave & 83rd St', 'Ewing Ave & 105th St', 'Marquette Ave & 79th St', 'Lafayette Ave & 95th St']
    rents_station_list = [element for element in Variables.final_station_list if element not in errors]
    rents_series = pd.DataFrame(Variables.rent_clusters, index=rents_station_list)

    rents = rents.assign(cluster=rents_series)

    rent_cluster_0_avg = rents[rents['cluster'] == 0].mean(axis=0)
    rent_cluster_1_avg = rents[rents['cluster'] == 1].mean(axis=0)
    rent_cluster_2_avg = rents[rents['cluster'] == 2].mean(axis=0)
    rent_cluster_3_avg = rents[rents['cluster'] == 3].mean(axis=0)

    avg_df = pd.concat({'cluster_0': rent_cluster_0_avg,
                        'cluster_1': rent_cluster_1_avg,
                        'cluster_2': rent_cluster_2_avg,
                        'cluster_3': rent_cluster_3_avg}, axis=1)
    avg_df = avg_df[:-1]

    for column in avg_df:
        sns.lineplot(data=avg_df[column])
        plt.show()

def visualize_returns():
    returns = pd.read_csv('../Results/return_df_normalized.csv', index_col=0)
    errors = ['Ewing Ave & 106th St', 'Parkside Ave & Armitage Ave', 'S Aberdeen St & W 106th St']
    returns_station_list = [element for element in Variables.final_station_list if element not in errors]
    returns_series = pd.DataFrame(Variables.return_clusters, index=returns_station_list)
    returns = returns.assign(cluster=returns_series)

    return_cluster_0_avg = returns[returns['cluster'] == 0].mean(axis=0)
    return_cluster_1_avg = returns[returns['cluster'] == 1].mean(axis=0)
    return_cluster_2_avg = returns[returns['cluster'] == 2].mean(axis=0)
    return_cluster_3_avg = returns[returns['cluster'] == 3].mean(axis=0)


    avg_df = pd.concat({'cluster_0': return_cluster_0_avg,
                        'cluster_1': return_cluster_1_avg,
                        'cluster_2': return_cluster_2_avg,
                        'cluster_3': return_cluster_3_avg}, axis=1)
    avg_df = avg_df[:-1]

    for column in avg_df:
        sns.lineplot(data=avg_df[column])
        plt.show()

def calculate_leiden_communities():
    dataframe = pd.read_csv('../Results/adjacency_matrix_with_threshold.csv', index_col=0)
    A = np.array(dataframe)

    G = nx.from_numpy_matrix(A)
    network_communities = algorithms.leiden(G)
    nx.draw(G)
    plt.show()

def assemble_final_df():
    leiden_results_dict = dict()
    for n, element in enumerate(Variables.leiden_results):
        for sub_element in element:
            leiden_results_dict[sub_element] = n

    leiden_results_dict_final = dict()
    for key, value in sorted(leiden_results_dict.items()):
        leiden_results_dict_final[key] = value

    leiden_df = pd.DataFrame(data={'station_name': Variables.final_station_list,
                                   'leiden_community': leiden_results_dict_final.values()}, index=Variables.final_station_list)

    errors = ['Springfield Ave & 63rd St', 'Michigan Ave & 113th St', 'Colfax Ave & 83rd St', 'Ewing Ave & 105th St',
              'Marquette Ave & 79th St', 'Lafayette Ave & 95th St']
    rents_station_list = [element for element in Variables.final_station_list if element not in errors]
    rent_df = pd.DataFrame(data={'station_name': rents_station_list, 'rent_cluster': Variables.rent_clusters}, index=rents_station_list)

    errors = ['Ewing Ave & 106th St', 'Parkside Ave & Armitage Ave', 'S Aberdeen St & W 106th St']
    returns_station_list = [element for element in Variables.final_station_list if element not in errors]
    return_df = pd.DataFrame(data={'station_name': returns_station_list, 'rent_cluster': Variables.return_clusters},
                           index=returns_station_list)

    dfs = [leiden_df, rent_df, return_df]
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='station_name'), dfs)
    df_final = df_final.set_index('station_name')

    df_final['leiden_community'] = df_final['leiden_community'].replace(list(range(5,28)), list([5]*23))

    return df_final.to_csv('df_final.csv')

def strength_degree_correlation():
    df = pd.DataFrame(data={'strength': Variables.strength.values(), 'degree': Variables.degree.values()}, index=Variables.all_stations_list)

    plt.figure()
    plt.hist(list(df['degree']))
    plt.title('Degree distribution')
    plt.show()

    plt.figure()
    plt.hist(list(df['strength']))
    plt.title('Strength distribution')
    plt.show()

    plt.figure()
    plt.xlabel('Strength')
    plt.ylabel('Degree')
    plt.scatter(df['strength'], df['degree'])
    plt.show()

    results = stats.pearsonr(df['strength'], df['degree'])
    print(results)     # (0.8653897438721062, 2.351e-320)


if __name__ == "__main__":
    hierarchical_clustering()
    visualize_rents()
    visualize_returns()
    calculate_leiden_communities()
    assemble_final_df()
    strength_degree_correlation()