# PPG Heart Rate Classifier
SVM-based model for extracting heart rate signal segments from existing or realtime PPG (photoplethysmogram) measurements, which usually tend to be difficult to parse, noisy, and motion affected, particularly when originating from a sensor embedded in a wearable device. The model can be integrated into a mobile application to automatically detect heart rate patterns in live PPG stream (e.g. via BLE) and extract meaningful data for visualization.

The training set, located in `dataPPG/`, is comprised of anonymized PPG readings with corresponding timestamps, taken using multiple wearable devices from children at a Montessori preschool over the course of one week. Readings were obtained with an NJR reflective sensor and the popular TI [link](http://www.ti.com/product/AFE4400 "AFE4400") medical FE.

Input data should be raw PPG readings, such as this:

<div style="text-align:center" align="center"><img src="figures/before.png" width="500"><br />Long term PPG measurement</div>
<br />
The algorithm (`HRClassifier.py`) divides the data into smaller segments and produces a binary classification with clear HR ranges and the calculated beats-per-minute number commonly desired for each range (known as the heart rate variability). An example of a positively classified signal is shown below. 
<br />
<div style="text-align:center" align="center"><img src="figures/after.png" width="500"><br />Positively classified segment, zoomed in;<br />97.5 beats per minute</div>
