import matplotlib.pyplot as plt

def plot(rows, cols, main_title, df, df_cols):
    fig, axs = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axs[row, col].hist(df[df_cols[row][col]].values)
            axs[row, col].set_title(df_cols[row][col])

    plt.suptitle(main_title)
    plt.tight_layout()

    return fig, axs