import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skspatial.objects import Line
# from skspatial.objects import Point
from skspatial.plotting import plot_2d
from shapely.geometry import Point
from shapely.geometry import LineString


df = pd.read_csv("/Users/yliao13/PycharmProjects/phd_class/Machine_Learning_Class_8210/data_files/dataset_1.csv", header=0)

v1 = df['V1'].to_numpy()
v2 = df['V2'].to_numpy()
label = df['label'].to_numpy()

colors = {0: "red", 1: "blue"}

# plt.scatter(v1, v2, edgecolor="none", alpha=0.8)

plt.scatter(df['V1'].to_numpy(), df['V2'].to_numpy(), edgecolor="none", alpha=0.8, c=df['label'].map(colors))

line = Line.from_points([0, 0], [0.67510539, 0.7377213])
line2 = Line.from_points([0, 0], [-0.71381103, 1])

x_values = [0, 0.67510539 * 50]
y_values = [0, 0.7377213 * 50]
plt.plot(x_values, y_values)

x_values_2 = [0 * 5,  -0.71381103 * 5]
y_values_2 = [0 * 5, 1 * 5]
plt.plot(x_values_2, y_values_2)

x_values_2 = [0 * 5,  0.71381103 * 5]
y_values_2 = [0 * 5, -1 * 5]
plt.plot(x_values_2, y_values_2, c='orange')

# line = Line.from_points([0, 0], [1, -0.71381103])
color = ['red', 'blue']

n=0

# def dot_projection(x, line):
#     point = Point(x[0], x[1])
#
#
#     x = np.array(point.coords[0])
#
#     u = np.array(line.coords[0])
#     v = np.array(line.coords[len(line.coords) - 1])
#
#     n = v - u
#     n /= np.linalg.norm(n, 2)
#
#     P = u + n * np.dot(x - u, n)
#     return P

# for index, item in df.iterrows():
#     # point = Point(item.iloc[0:2].to_numpy())
#     # point_projected = line.project_point(point)
#     line = LineString([(0, 0), (0.67510539, 0.7377213)])
#     point_projected = dot_projection(item.iloc[0:2].to_numpy(), line)
#
#     if item.iloc[2] == 1:
#         dotColor = color[1]
#     if item.iloc[2] == 0:
#         dotColor = color[0]
#     plt.scatter(point_projected[0], point_projected[1], c=dotColor, edgecolor="none", s=70)
#
#
# for index, item in df.iterrows():
#     # point = Point(item.iloc[0:2].to_numpy())
#     # point_projected = line2.project_point(point)
#     line = LineString([(0, 0), (-0.71381103, 1)])
#     point_projected = dot_projection(item.iloc[0:2].to_numpy(), line)
#
#     if item.iloc[2] == 1:
#         dotColor = color[1]
#     if item.iloc[2] == 0:
#         dotColor = color[0]
#     plt.scatter(point_projected[0], point_projected[1], c=dotColor, edgecolor="none", s=70)


plt.xlabel("x")
plt.ylabel("y")
labels = ["PC1", 'W']
plt.legend(labels)
plt.show()