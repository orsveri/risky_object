import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_filepaths, labels, metric="AP"):
    """
    Plots the specified metric from multiple CSV files.

    Parameters:
        csv_filepaths (list of str): List of paths to CSV files.
        labels (list of str): List of labels corresponding to each CSV file for the legend.
        metric (str): The metric to plot (default is "AP").
    """
    plt.figure(figsize=(10, 6))  # Set the size of the plot

    for csv_file, label in zip(csv_filepaths, labels):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Plot the specified metric with epoch on the x-axis
        plt.plot(df["epoch"], df[metric], label=label)

    # Add labels, title, and legend to the plot
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} || Test on ROL")
    plt.legend()
    plt.grid(True)  # Optional: Adds a grid for better readability

    # Show the plot
    plt.show()


# Example usage:
csv_filepaths = [
    'logs/2_stage/only_rol/version_2/results/rol/test_rol.csv',
    'logs/2_stage/only_dota/version_2/results/rol/test_rol.csv',
    'logs/2_stage/rol_dota/version_2/results/rol/test_rol.csv',
    'logs/2_stage/rol_gta_800/version_2/results/rol/test_rol.csv'
]

labels = [
    'only_rol',
    'only_dota',
    'rol_dota',
    'rol_gta_800'
]

# Call the function to plot the "AP" metric
plot_metrics(csv_filepaths, labels, metric="AP")
