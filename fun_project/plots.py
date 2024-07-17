import matplotlib.pyplot as plt

# defining labels
activities = ['', '', '', '']

# portion covered by each label
slices = [3, 7, 8, 6]

# color for each label
colors = ['r', 'y', 'g', 'b']

# plotting the pie chart
# plt.pie(slices, labels=activities, colors=colors,
#         startangle=90, shadow=True, explode=(0, 0, 0.1, 0),
#         radius=1.2, autopct='%1.1f%%')
plt.pie(slices, labels=activities, colors=colors,
        startangle=90, shadow=True, explode=(0, 0, 0.1, 0),
        radius=1.2)

# plotting legend
# plt.legend()

# showing the plot
plt.savefig('/Users/yliao13/Desktop/data_analysis_logo.png', dpi=300)