# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame(
    {
        "lakeid": [
            298315,
            298316,
            298317,
        ],
        "eq1_r2": [0, 1, 2],
        "eq1_rmse": [10, 400, 100],
        "eq1_mae": [0, 43, 1000],
    }
)


# normalize column
def df_with_normalized_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


df = df_with_normalized_column(df, "eq1_r2")
df = df_with_normalized_column(df, "eq1_rmse")
df = df_with_normalized_column(df, "eq1_mae")
print(df)

# number of variable
columns_to_plot = ["eq1_r2", "eq1_rmse", "eq1_mae"]
N = len(columns_to_plot)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], columns_to_plot)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0, 0.5, 1], ["0.1", "0.5", "1.0"], color="grey", size=7)
plt.ylim(0, 1)


def plot_lake(ax, df, lakeid, columns_to_plot, fill_color):
    lake_row = df[df["lakeid"] == lakeid]
    lake_row = lake_row[columns_to_plot]

    values = lake_row.values.tolist()[0]

    print(values)
    # add first value back so polygon loops around
    values += values[0:1]

    # ax.plot(angles, values, linewidth=1, linestyle="solid", label=f"lake{lakeid}")
    ax.fill(angles, values, fill_color, alpha=0.1)


plot_lake(ax, df, 298316, columns_to_plot, "r")
plot_lake(ax, df, 298317, columns_to_plot, "b")

plt.show()
