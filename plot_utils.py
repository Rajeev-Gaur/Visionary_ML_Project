import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Diagonal line for reference
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')

    # Save the plot in the output directory
    plt.savefig('output/predictions_plot.png')  
    plt.close()  # Close the figure to avoid display in interactive environments
