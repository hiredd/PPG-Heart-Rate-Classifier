import csv
import serial
import sys
import numpy as np
import scipy as sp
from scipy import signal
from sklearn import svm
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.model_selection import train_test_split
sys.path.append('libs')
import detect_peaks
from HRClassifier import HRClassifier

c = HRClassifier()

c.train()

if 1==0:
	# Load test set
	PPGdata = c.getPPGDataFromFile("6_1_4")

	results = c.getValidHRranges(PPGdata)

	for i in results:
	    print(i[0]) # print heart rate value

# Live dataset

ser = serial.Serial(port='COM4', baudrate=9600)
while True:
	PPGdata = c.getPPGDataFromStream(ser)
	results = c.getValidHRranges(PPGdata)
	for i in results:
		print(i[0]) # print heart rate value
