import numpy as np
from collections import defaultdict

def calculate_accuracy(y_true, output_y):
    if len(y_true.shape) > 1:
        correct_predictions = np.argmax(output_y, axis = 1) == np.argmax(y_true, axis = 1)
    else:
        correct_predictions = output_y == y_true
    num_correct = np.sum(correct_predictions)
    return (num_correct / len(y_true)) * 100.0

def calculate_class_based_accuracy(y_true, output_y): 
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis = 1)
        output_y = np.argmax(output_y, axis = 1)

    correct_predictions = defaultdict(int)  
    total_predictions = defaultdict(int)

    for true_label, pred_label in zip(y_true, output_y):
        correct_predictions[true_label] += (true_label == pred_label)
        total_predictions[true_label] += 1

    accuracies = defaultdict(float, value = 100.0) 
    for label in correct_predictions:
        accuracies[label] = (correct_predictions[label] / total_predictions[label]) * 100.0
    return accuracies