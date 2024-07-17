import argparse

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt

def pca_analysis_and_plot(input_file, pc_number):
    df = pd.read_csv(input_file, header=None, sep='\s+')

    x = df.loc[:, :].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=int(pc_number))
    principal_components = pca.fit_transform(x)

    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    # # for K=2 PCA plot using matplotlib.pyplot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(principal_components[:, 0], principal_components[:, 1], )
    # plt.xlabel(labels['0'])
    # plt.ylabel(labels['1'])
    # plt.title("PCA analysis (2D plot) (K=2)")
    # plt.show()


    # 2D plot for PCA
    fig_2D = px.scatter_matrix(principal_components, labels=labels, dimensions=range(int(pc_number)))

    fig_2D.update_traces(diagonal_visible=False)
    fig_2D.show()

    # 3D plot for PCA
    if int(pc_number) == 3:
        total_var = pca.explained_variance_ratio_.sum() * 100
        fig_3D = px.scatter_3d(principal_components, x=0, y=1,z=2, title=f'Total Explained Variance: {total_var:.2f}%',
                               labels=labels)
        fig_3D.show()



def main():
    parser = argparse.ArgumentParser(description="Reads factors for each sample and writes them into CSV file")
    parser.add_argument('--input_file', help='files contains matrix, either csv/txt, separate with spaces.',
                        required=True)
    parser.add_argument('--principal_components_number', help='integer number for the value of K', required=True)

    args = parser.parse_args()

    input_file = args.input_file
    pc_number = args.principal_components_number

    pca_analysis_and_plot(input_file, pc_number)


if __name__ == '__main__':
    main()
