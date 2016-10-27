# PPG-heart-rate-classifier
Extract quality heart rate segments from noisy/motion affected raw PPG measurements. Once trained, the model can be integrated into a mobile application to automatically detect heart rate patterns in live PPG stream (e.g. via BLE) and calculate the person's BPM number.

See _thesis_excerpt.pdf_ and _HRClassifier.py_ for details

Go from this:

<div style="text-align:center" align="center"><img src="figures/before.png" width="350"><br />Long term PPG measurement</div>

to this:

<div style="text-align:center" align="center"><img src="figures/after.png" width="350"><br />Positively classified segment, zoomed in<br />97.5 beats per minute</div>
