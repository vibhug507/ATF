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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

        self.__most_affected_classes = []
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
    
    def get_most_affected_classes(self, limit = 5):
        class_accuracies_sum = defaultdict(float)
        class_accuracies_cnt = defaultdict(int)
        for attack, results in self.__results.items():
            if attack == 'original':
                continue
            for acc_dict in results['class_accuracies']:
                print(acc_dict)
                for label, acc in acc_dict.items():
                    class_accuracies_sum[label] += acc
                    class_accuracies_cnt[label] += 1
        accuracies = []
        for label, acc_sum in class_accuracies_sum.items():
            accuracies.append((acc_sum / class_accuracies_cnt[label], label))
        accuracies = sorted(accuracies, key=lambda x: (x[0], x[1]))
        self.__most_affected_classes = []
        for i in range(min(limit, len(accuracies))):
            self.__most_affected_classes.append(accuracies[i][1])
        return self.__most_affected_classes

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
        styles = getSampleStyleSheet()
        elements = []

        # Add heading
        heading_style = ParagraphStyle(name='Heading1', fontName='Helvetica-Bold', fontSize=16, alignment=1, spaceAfter=12)
        heading_text = "<u>ATF Results Analysis</u>"
        heading = Paragraph(heading_text, heading_style)
        elements.append(heading)
        elements.append(Spacer(1, 12))

        # Comparison Table
        comparison_data = [['Attack', 'Avg Accuracy', 'Attack Metric']]
        for attack, results in self.__results.items():
            if attack == 'original':
                continue
            avg_acc = np.mean(results['accuracy'])
            attack_metric = attack_metrics.get(attack, 'N/A')
            comparison_data.append([attack, f"{avg_acc:.2f}%", attack_metric])
        comparison_table = Table(comparison_data, colWidths=[100, 100, 100, 200])
        comparison_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(comparison_table)

        # Add most affected classes
        most_affected_classes = self.get_most_affected_classes()
        most_affected_classes_text = f"Most Affected Classes: {', '.join(str(label) for label in most_affected_classes)}"
        most_affected_classes_paragraph = Paragraph(most_affected_classes_text, styles['Normal'])
        elements.append(most_affected_classes_paragraph)
        elements.append(Spacer(1, 12))

        # Add attack results
        for attack, results in self.__results.items():
            if attack == 'original':
                continue
            attack_title = Paragraph(f"<b>Attack: {attack.capitalize()}</b>", styles['Heading2'])
            elements.append(attack_title)
            elements.append(Spacer(1, 12))

            attack_accuracy_text = f"Average Accuracy after Attack: {np.mean(results['accuracy']):.2f}%"
            attack_accuracy = Paragraph(attack_accuracy_text, styles['Normal'])
            elements.append(attack_accuracy)
            elements.append(Spacer(1, 12))

            attack_data = [['Attack', 'Metric', 'Set of Values']]
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

            attack_data = [['Metric Vaue', 'Accuracy']]
            attack_metric = attack_metrics[attack]
            for metric, acc in zip(results[attack_metric], results['accuracy']):
                attack_data.append([metric, f"{acc:.2f}%"])
            attack_table = Table(attack_data, colWidths=[100, 100])
            attack_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                            ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
            elements.append(attack_table)
            elements.append(Spacer(1, 12))

            # Graph for accuracies vs metric values
            sorted_metrics = sorted(results[attack_metric])
            sorted_accuracies = [acc for _, acc in sorted(zip(results[attack_metric], results['accuracy']))]
            plt.figure(figsize=(8, 6))
            plt.plot(sorted_metrics, sorted_accuracies, marker='o')
            plt.xlabel(attack_metric.capitalize())
            plt.ylabel('Accuracy (%)')
            plt.title(f'{attack.capitalize()} Attack: Accuracies vs {attack_metric.capitalize()}')
            plt.grid(True)
            plt.tight_layout()
            graph_filename = f'{attack}_graph.png'
            plt.savefig(graph_filename)
            plt.close()

            # Add the graph to PDF
            graph_image = Image(graph_filename, width=400, height=300)
            elements.append(graph_image)
            elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)