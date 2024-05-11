from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
import numpy as np
from utils.metrics import *
from attacks.classification_attack import *

class PGD(ClassificationAttack): 
    def __init__(self, model, test_x, test_y, eps = 0.1, norm = np.inf, eps_iter = 0.01, nb_iter = 31):
        super().__init__(model, test_x, test_y)
        self.eps = eps
        self.norm = norm
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter

        self.result_object = {'eps': set([0.01, 0.05, 0.1, 0.5, self.eps if self.eps >= eps_iter else 0.01]),
                              'accuracy': [],
                              'class_accuracies': []}

        for eps in self.result_object['eps']:
            adversarial_input = projected_gradient_descent(model, test_x, eps, self.eps_iter, self.nb_iter, self.norm)
            adversarial_output = model(adversarial_input)
            adversarial_accuracy = calculate_accuracy(test_y, adversarial_output)
            adversarial_class_accuracies = calculate_class_based_accuracy(test_y, adversarial_output)
            self.result_object['accuracy'].append(adversarial_accuracy)
            self.result_object['class_accuracies'].append(adversarial_class_accuracies)