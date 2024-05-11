import numpy as np
from utils.metrics import *
from attacks.classification_attack import *
from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack
from foolbox import TensorFlowModel
from foolbox.criteria import Misclassification
import tensorflow as tf
import math

class SaltAndPepperNoise(ClassificationAttack):
    def __init__(self, model, test_x, test_y, steps = 200):
        super().__init__(TensorFlowModel(model, bounds = (np.min(test_x), np.max(test_x))), test_x, test_y)
        self.steps = steps
        labels = np.argmax(test_y, axis = 1)

        self.result_object = {'steps': [int(math.ceil(steps * x)) for x in [0.25, 0.50, 0.75, 1.0]],
                              'accuracy': [],
                              'class_accuracies': []}
        inputs = tf.convert_to_tensor(test_x)
        criterion = Misclassification(tf.convert_to_tensor(labels))
        
        for step in self.result_object['steps']:
            attack_obj = SaltAndPepperNoiseAttack(steps = step)
            adversarial_input = attack_obj.run(self.model, criterion = criterion, inputs = inputs)
            adversarial_output = model(adversarial_input)
            adversarial_accuracy = calculate_accuracy(test_y, adversarial_output)
            adversarial_class_accuracies = calculate_class_based_accuracy(test_y, adversarial_output)
            self.result_object['accuracy'].append(adversarial_accuracy)
            self.result_object['class_accuracies'].append(adversarial_class_accuracies)