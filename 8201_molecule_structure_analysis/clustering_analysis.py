import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


def hierarchical_clustering(input_file):
    df = pd.read_csv(input_file, header=None, sep='\s+')
    df.to_csv("/Users/ericliao/Desktop/ecoli_matrix.csv", sep='\t', encoding='utf-8', index=False)
    df_scaled = normalize(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dendrograms_plot = shc.dendrogram(shc.linkage(df_scaled, method='ward'))
    # plt.show()
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    cluster.fit(df_scaled)
    labels = cluster.labels_

    plt.scatter(df_scaled[labels == 0], df_scaled[labels == 0], s=50, marker='o', color='red')
    plt.scatter(df_scaled[labels == 1], df_scaled[labels == 1], s=50, marker='o', color='yellow')
    plt.scatter(df_scaled[labels == 2], df_scaled[labels == 2], s=50, marker='o', color='orange')
    plt.scatter(df_scaled[labels == 3], df_scaled[labels == 3], s=50, marker='o', color='blue')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Reads factors for each sample and writes them into CSV file")
    parser.add_argument('--input_file', help='files contains matrix, either csv/txt, separate with spaces.',
                        required=True)

    args = parser.parse_args()
    input_file = args.input_file
    hierarchical_clustering(input_file)


if __name__ == '__main__':
    main()