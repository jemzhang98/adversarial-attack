"""
The script demonstrates a simple example of using ART with scikit-learn. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import skimage
import numpy as np
import pickle
import os.path

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import SimBA
from art.estimators.classification import SklearnClassifier
from art.utils import load_mnist
from art.utils import load_dataset
from skimage import color
from skimage.feature import hog
from sklearn.svm import SVC
from numpy import save


def transform2Grey(input_image):
    """perform the transformation and return an array"""
    return np.array([create_features(img) for img in input_image])

def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = skimage.color.rgb2gray(img)
    # get HOG features from greyscale image
    # hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features

filename = 'Models/sklearn_cifar10_full_svm_model.sav'

# Step 1: Load the MNIST dataset
print('Loading dataset')

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset("cifar10")
# x_train, y_train = x_train[:1000], y_train[:1000]
x_train_g = transform2Grey(x_train)
# x_test, y_test = x_test[:100], y_test[:100]
x_test_g = transform2Grey(x_test)

# Step 1a: Flatten MNIST dataset
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# nb_samples_train = x_train.shape[0]
# nb_samples_test = x_test.shape[0]
# x_train = x_train.reshape((nb_samples_train, 28 * 28))
# x_test = x_test.reshape((nb_samples_test, 28 * 28))

# Load the model from disk if already made
if os.path.isfile('./' + filename):
    print('Loading model')
    classifier = pickle.load(open(filename, 'rb'))
else:
    print('Training dataset')
    # Step 2: Create the model
    model = SVC(C=1.0, kernel="rbf")

    # Step 3: Create the ART classifier
    classifier = SklearnClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

    # Step 4: Train the ART classifier
    classifier.fit(x_train, y_train)

    # Save the model to disk
    pickle.dump(classifier, open(filename, 'wb'))


# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test_g)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
attack = SimBA(classifier=classifier, max_iter=1000)
x_test_adv = attack.generate(x=x_test)

print('Saving generated adv data')
save('cifar10_uni_fgsm_adv.npy', x_test_adv)

# Step 7: Evaluate the ART classifier on adversarial test examples
x_test_adv_flat = transform2Grey(x_test_adv)
predictions = classifier.predict(x_test_adv_flat)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
