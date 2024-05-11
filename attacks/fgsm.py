from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import numpy as np
from utils.metrics import *
from attacks.classification_attack import *

class FGSM(ClassificationAttack): 
    def __init__(self, model, test_x, test_y, eps = 0.1, norm = np.inf):
        with open("print_file.txt",'w') as f:
            print("opened")

        super().__init__(model, test_x, test_y)
        self.eps = eps
        self.norm = norm

        self.result_object = {'eps': set([0.001, 0.01, 0.05, 0.1, 0.5, self.eps]),
                              'accuracy': [],
                              'class_accuracies': []}
        print("hellllloooooo")
        for eps in self.result_object['eps']:
            adversarial_input = fast_gradient_method(model, test_x, eps, self.norm)
            print("done1")
            adversarial_output = model(adversarial_input)
            print("done2")
            adversarial_accuracy = calculate_accuracy(test_y, adversarial_output)
            print("done3")
            adversarial_class_accuracies = calculate_class_based_accuracy(test_y, adversarial_output)
            print("done4")
            self.result_object['accuracy'].append(adversarial_accuracy)
            self.result_object['class_accuracies'].append(adversarial_class_accuracies)