import numpy as np
from utils.metrics import *
from attacks.classification_attack import *
from foolbox.attacks.carlini_wagner import L2CarliniWagnerAttack
from foolbox import TensorFlowModel
from foolbox.criteria import Misclassification
import tensorflow as tf

class CarliniWagner(ClassificationAttack):
    def __init__(self, model, test_x, test_y, confidence = 0.0, max_iter = 10):
        super().__init__(TensorFlowModel(model, bounds = (np.min(test_x), np.max(test_x))), test_x, test_y)
        self.confidence = confidence
        self.max_iter = max_iter

        labels = np.argmax(test_y, axis = 1)

        self.result_object = {'confidence': set([0.0, 0.3, 0.6, 0.9, 1.0, self.confidence]),
                              'accuracy': [],
                              'class_accuracies': []}
        inputs = tf.convert_to_tensor(test_x)
        criterion = Misclassification(tf.convert_to_tensor(labels))
        
        for confidence in self.result_object['confidence']:
            attack_obj = L2CarliniWagnerAttack(binary_search_steps = 2, steps = max_iter, confidence = confidence)
            adversarial_input = attack_obj.run(self.model, criterion = criterion, inputs = inputs)
            adversarial_output = model(adversarial_input)
            adversarial_accuracy = calculate_accuracy(test_y, adversarial_output)
            adversarial_class_accuracies = calculate_class_based_accuracy(test_y, adversarial_output)
            self.result_object['accuracy'].append(adversarial_accuracy)
            self.result_object['class_accuracies'].append(adversarial_class_accuracies)