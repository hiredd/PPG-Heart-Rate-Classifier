import csv
import numpy as np
import scipy as sp
from scipy import signal
from sklearn import svm
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import detect_peaks
from HRClassifier import HRClassifier

c = HRClassifier()

c.train()

# Load test set
PPGdata = c.getPPGData("6_1_4")

results = c.getValidHRranges(PPGdata)

for i in results:
	print(i[0]) # print heart rate value
