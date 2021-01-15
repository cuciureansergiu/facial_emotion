from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets, svm, metrics
from matplotlib.pyplot import imread
from fnmatch import fnmatch as match
from zipfile import ZipFile
from util import rgb2gray
from cv2 import resize
import wget, os, re
from hog import hog
import numpy as np
import progressbar
import pickle
import sys

print("Training facial emotion model")

if (os.path.isdir('./KDEF_and_AKDEF') == False):
    print("Getting KDEF database")
    url = 'https://www.kdef.se/download/KDEF_and_AKDEF.zip'
    filename = wget.download(url)
    with ZipFile('KDEF_and_AKDEF.zip', 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall('./')

data_path = './KDEF_and_AKDEF/KDEF/'

dictionary = {
    "AF" : "afraid",
    "AN" : "angry",
    "DI" : "disgusted",
    "HA" : "happy",
    "NE" : "neutral",
    "SA" : "sad",
    "SU" : "surprised"
}

img_w = 512
img_h = 512

hog = hog()
hog.summary()

timecount = 0
avg = 0
# 140 actors * 7  emotions
bar = progressbar.ProgressBar(maxval=140*7, \
    widgets=[progressbar.Bar('=', '[', ']'),\
    ' ', progressbar.ETA(),' ', progressbar.Percentage()])
bar.start()

labels = []
data = np.zeros((1, int(hog.feature_size * 64))) # Create empty image so the other can stack here

print("Reading the input data")
for person in os.listdir(data_path):
    for image in os.listdir(data_path + person):
        # Only front facing profiles
        if match(image, '*S.JPG'):

            keyword = re.match(r"([a-z]+)([0-9]+)([a-z]+)", image, re.I)
            category = keyword.groups()[2]
            category = category[:-1]
            img = imread(data_path + person + "/" + image)
            img = resize(img, (img_w, img_h))
            img = rgb2gray(img)
            feature = []
            for i in range(0, img_w, 64):
                for j in range(0, img_h, 64):
                    patch = img[i:i+64, j:j+64]
                    feature = np.concatenate((feature, [hog.get_feature(patch)]), axis=None)
            data = np.vstack((data,feature))
            labels.append(list(dictionary).index(category))
            bar.update(timecount := timecount + 1)
data = data[1:] # Delete the dummy feature
labels = np.array(labels)
bar.finish()

classifier = svm.SVC()
data_train, data_test, labels_train, label_test = train_test_split(
    data, labels, test_size=0.1, shuffle=False)

print("Training the model")
classifier.fit(data_train, labels_train)
pickle.dump(classifier, open('emotion.svm', 'wb'))

print("Testing the model")
predicted = classifier.predict(data_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(label_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, data_test, label_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

#Save the confusion matrix to a file
f = open("report.txt", "a")
f.write("Classification report for classifier\n")
f.write(str(metrics.classification_report(label_test, predicted)))
f.write(str(confusion_matrix(label_test, predicted)))
f.close()
