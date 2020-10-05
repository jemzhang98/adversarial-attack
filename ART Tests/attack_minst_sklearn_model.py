"""
The script demonstrates a simple example of using ART with scikit-learn. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from sklearn.svm import SVC
import numpy as np
import  pickle
import os.path

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier
from art.utils import load_mnist

filename = 'Models/sklearn_mnist_svm_model.sav'

# Step 1: Load the MNIST dataset
print('Loading dataset and model')
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Flatten dataset
nb_samples_train = x_train.shape[0]
nb_samples_test = x_test.shape[0]
x_train = x_train.reshape((nb_samples_train, 28 * 28))
x_test = x_test.reshape((nb_samples_test, 28 * 28))

# Reduce the dataset
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]

# load the model from disk
if os.path.isfile('./' + filename):
    classifier = pickle.load(open(filename, 'rb'))
else:
    # Step 2: Create the model
    model = SVC(C=1.0, kernel="rbf")

    # Step 3: Create the ART classifier
    classifier = SklearnClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

    # Step 4: Train the ART classifier
    classifier.fit(x_train, y_train)

    # Save the model to disk
    pickle.dump(classifier, open(filename, 'wb'))


# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
attack = ZooAttack(classifier=classifier)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
