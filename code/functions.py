import matplotlib.pyplot as plt

def plot(rows, cols, main_title, df, df_cols):
    fig, axs = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axs[row, col].hist(df[df_cols[row][col]].values)
            axs[row, col].set_title("_".join(df_cols[row][col].split("_")[1:]))

    plt.suptitle(main_title)
    plt.tight_layout()