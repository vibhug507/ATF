import h5py
import tensorflow as tf
import numpy as np
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.carlini_wagner import CarliniWagner
from attacks.deepfool import DeepFool
from attacks.salt_and_pepper_noise import SaltAndPepperNoise
from utils.metrics import *
from prettytable import PrettyTable
from keras import backend as K
import keras
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer
from reportlab.lib import colors

attacks = ['fgsm', 'pgd', 'carlini_wagner', 'deepfool', 'sp_noise']
attack_metrics = {'fgsm': 'eps', 'pgd': 'eps', 'carlini_wagner': 'confidence', 
                  'deepfool': 'overshoot', 'sp_noise': 'steps'}

class Net(tf.keras.Model):
    '''
    param model: a tensorflow model
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, x):
        return self.model(x)

class ATF:
    '''
    param model: h5 model file (.h5)
    param X: input file (.h5)
    param Y: output file (.h5)
    param attacks_to_run: attacks for which the model will be tested (list)
    '''
    def __init__(self, model = None, X = None, Y = None, attacks_to_run = attacks):
        K.clear_session()
        self.__model = Net(keras.models.load_model(model))
        with h5py.File(X, 'r') as f:
            self.__X = f['images'][:]
        with h5py.File(Y, 'r') as f:
            self.__Y = f['output'][:]
        self.__attacks_to_run = attacks_to_run

        self.__original_output = self.__model(self.__X)
        self.__original_accuracy = calculate_accuracy(self.__Y, self.__original_output)
        self.__original_class_accuracies = calculate_class_based_accuracy(self.__Y, self.__original_output)

        self.__results = {'original': {'accuracy': self.__original_accuracy, 
                                     'class_accuracies': self.__original_class_accuracies}}
        
        self.__attack_methods = {'fgsm': self.__perform_fgsm,
                                 'pgd': self.__perform_pgd,
                                 'carlini_wagner': self.__perform_cw,
                                 'deepfool': self.__perform_deepfool,
                                 'sp_noise': self.__perform_sp_noise}

    def run(self, attacks_to_run = None):
        if attacks_to_run is None:
            attacks_to_run = self.__attacks_to_run
        for attack in attacks_to_run: 
            if attack in self.__attack_methods:
                self.__attacks_to_run.append(attack)
                if attack not in self.__results:
                    self.__attack_methods[attack]()
        return self.__results
    
    def get_results(self):
        return self.__results
        
    def __perform_fgsm(self, eps = 0.1, norm = np.inf):
        if 'fgsm' not in self.__results:
            fgsm_obj = FGSM(self.__model, self.__X, self.__Y, eps, norm)
            self.__results['fgsm'] = fgsm_obj.result_object
    
    def __perform_pgd(self, eps = 0.1, eps_iter = 0.01, nb_iter = 31, norm = np.inf):
        if 'pgd' not in self.__results:
            pgd_obj = PGD(self.__model, self.__X, self.__Y, eps, norm, eps_iter, nb_iter)
            self.__results['pgd'] = pgd_obj.result_object
    
    def __perform_cw(self, confidence = 0.0, max_iter = 10):
        if 'carlini_wagner' not in self.__results:
            carlini_wagner_obj = CarliniWagner(self.__model, self.__X, self.__Y, confidence, max_iter)
            self.__results['carlini_wagner'] = carlini_wagner_obj.result_object

    def __perform_deepfool(self, steps = 20, overshoot = 0.02):
        if 'deepfool' not in self.__results:
            deepfool_obj = DeepFool(self.__model, self.__X, self.__Y, steps, overshoot)
            self.__results['deepfool'] = deepfool_obj.result_object
            
    def __perform_sp_noise(self, steps = 20):
        if 'sp_noise' not in self.__results:
            sp_noise_obj = SaltAndPepperNoise(self.__model, self.__X, self.__Y, steps)
            self.__results['sp_noise'] = sp_noise_obj.result_object

    def print_results(self):
        combined_table = PrettyTable()
        combined_table.field_names = ["Attack", "Min accuracy"]
        for attack in self.__results:
            if attack == 'original':
                continue
            print(attack)
            attack_table = PrettyTable()
            min_accuracy, index = 100.0, 0
            attack_metric = attack_metrics[attack]
            attack_table.field_names = [attack_metric, "After attack accuracy"]
            for metric in self.__results[attack][attack_metric]:
                attack_table.add_row([metric, self.__results[attack]['accuracy'][index]])
                min_accuracy = min(min_accuracy, self.__results[attack]['accuracy'][index])
                index += 1
            combined_table.add_row([attack, min_accuracy])
            print(attack_table)
        print(combined_table)
    
    def generate_results_pdf(self, filename='atf_results.pdf'):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []

        # Original Accuracy Table
        original_accuracy_data = [['Original Accuracy', f"{self.__original_accuracy:.2f}%"]]
        original_accuracy_table = Table(original_accuracy_data, colWidths=[200, 100])
        elements.append(original_accuracy_table)
        elements.append(Spacer(1, 12))

        # Original Class Accuracies Table
        original_class_accuracy_data = [['Class', 'Accuracy']]
        for label, acc in self.__original_class_accuracies.items():
            original_class_accuracy_data.append([f"Class {label}", f"{acc:.2f}%"])
        original_class_accuracy_table = Table(original_class_accuracy_data, colWidths=[100, 100])
        elements.append(original_class_accuracy_table)
        elements.append(Spacer(1, 12))

        comparison_data = [['Attack', 'Original Accuracy', 'Min Accuracy', 'Attack Metric']]
        for attack, results in self.__results.items():
            if attack == 'original':
                continue
            original_acc = self.__original_accuracy
            avg_acc = np.mean(results['accuracy'])
            attack_metric = results['attack_metric']
            comparison_data.append([attack, f"{original_acc:.2f}%", f"{avg_acc:.2f}%", attack_metric])
        comparison_table = Table(comparison_data, colWidths=[100, 100, 100, 200])
        comparison_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(comparison_table)

        # Attack Results Tables
        for attack, results in self.__results.items():
            if attack == 'original':
                continue
            attack_data = [['Attack', 'Metric', 'Value']]
            for metric, value in results.items():
                if metric == 'accuracy' or metric == 'class_accuracies':
                    continue
                attack_data.append([attack, metric, str(value)])  # Convert value to string
            attack_table = Table(attack_data, colWidths=[100, 100, 200])
            attack_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
            elements.append(attack_table)
            elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)