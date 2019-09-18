import glob
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

records = sorted(glob.glob('path-to-data/ecg-id-database-mod/Person_*/rec_*.png'))
model = load_model('model_ecg.h5')
w, h = 144, 224

person_1 = np.array(Image.open(records[0]).resize((h, w)))[:,:,:3] / 255
person_2 = np.array(Image.open(records[100]).resize((h, w)))[:,:,:3] / 255
person_3 = np.array(Image.open(records[200]).resize((h, w)))[:,:,:3] / 255
person_3_test = np.array(Image.open(records[201]).resize((h, w)))[:,:,:3] / 255

prob = model.predict([person_1.reshape((1, w, h, 3)), person_3_test.reshape((1, w, h, 3))])
pred = (prob>0.5)[0][0]
if pred:
  print("Person verified [confidence: {:.2f}%]".format(100*prob[0][0]))
else: 
  print("Wrong person - [confidence: {:.2f}%]".format(100*(1-prob[0][0])))