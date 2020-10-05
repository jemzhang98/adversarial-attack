import numpy as np 

path = './svm_reduced_testing_label_data.npy'

label_data = np.load(path)
print(label_data[:20])

#['i4', 'pl30', 'pl80', 'w57']


