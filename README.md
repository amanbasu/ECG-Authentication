# ECG-Authentication

![](https://github.com/amanbasu/ECG-Authentication/blob/master/images/result.png)

ECG is the depiction of the electric signals that come from the expansion and contraction of heart muscles, indirectly the flow of blood inside the heart. It depends on the anatomy and physiology of the heart which can vary with age, gender and myriad of other factors. And the most fascinating part is that it doesn't change over time for an individual, not even when the heart beats increase or decrease. 

This repository contains the code of developing a Deep Learning model to identify individuals based on their ECG signals. Its has been developed using Keras and TensorFlow and hosted on the Google Cloud Platform for deployment.

# Data

> The database contains 310 ECG recordings, obtained from 90 persons. Each recording contains:
> 
> ECG lead I, recorded for 20 seconds, digitized at 500 Hz with 12-bit resolution over a nominal ±10 mV range;
10 annotated beats (unaudited R- and T-wave peaks annotations from an automated detector);
information (in the .hea file for the record) containing age, gender and recording date.
> The records were obtained from volunteers (44 men and 46 women aged from 13 to 75 years who were students, colleagues, and friends of the author). The number of records for each person varies from 2 (collected during one day) to 20 (collected periodically over 6 months).
> 
> The raw ECG signals are rather noisy and contain both high and low-frequency noise components. Each record includes both raw and filtered signals:
> 
> Signal 0: ECG I (raw signal)
> Signal 1: ECG I filtered (filtered signal)
