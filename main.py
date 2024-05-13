import time
from atf import ATF

start_time = time.time()

atf = ATF('model.h5', 'input.h5', 'output.h5', ['fgsm', 'sp_noise', 'pgd', 'carlini_wagner', 'deepfool'])
results = atf.run(['fgsm', 'sp_noise', 'pgd', 'carlini_wagner', 'deepfool'])
atf.generate_results_pdf()

end_time = time.time()
print("Elapsed Time in second:", end_time - start_time)
print(atf.print_results())