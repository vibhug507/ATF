import numpy as np
from utils.metrics import *
from attacks.classification_attack import *
from foolbox.attacks.deepfool import DeepFoolAttack
from foolbox import TensorFlowModel
from foolbox.criteria import Misclassification
import tensorflow as tf

class DeepFool(ClassificationAttack):
    def __init__(self, model, test_x, test_y, steps = 20, overshoot = 0.02):
        super().__init__(TensorFlowModel(model, bounds = (np.min(test_x), np.max(test_x))), test_x, test_y)
        self.steps = steps
        self.overshoot = overshoot
        self.candidates = 10

        labels = np.argmax(test_y, axis = 1)

        self.result_object = {'overshoot': set([0.02, 0.05, 0.1, 1.0, self.confidence]),
                              'accuracy': [],
                              'class_accuracies': []}
        inputs = tf.convert_to_tensor(test_x)
        criterion = Misclassification(tf.convert_to_tensor(labels))
        
        for overshoot in self.result_object['overshoot']:
            attack_obj = DeepFoolAttack(steps = steps, overshoot = overshoot, candidates = self.candidates)
            adversarial_input = attack_obj.run(self.model, criterion = criterion, inputs = inputs)
            adversarial_output = model(adversarial_input)
            adversarial_accuracy = calculate_accuracy(test_y, adversarial_output)
            adversarial_class_accuracies = calculate_class_based_accuracy(test_y, adversarial_output)
            self.result_object['accuracy'].append(adversarial_accuracy)
            self.result_object['class_accuracies'].append(adversarial_class_accuracies)