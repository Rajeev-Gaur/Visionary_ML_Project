import sys
print(sys.path)

from plot_utils import plot_predictions

# Test with dummy data
y_true = [1, 2, 3, 4, 5]
y_pred = [1, 2, 3, 4, 5]

plot_predictions(y_true, y_pred)

