import time
from atf import ATF

atf = ATF('mobilenet_v3.h5', 'preprocessed_images.h5', 'output_classes.h5', ['fgsm'])
start_time = time.time()
print("main")
results = atf.run(['fgsm'])
end_time = time.time()
print("Elapsed Time in second:", end_time - start_time)
print(atf.print_results())