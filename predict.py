import glob
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

records = sorted(glob.glob('path-to-data/ecg-id-database-filter/Person_*/rec_*.png'))
model = load_model('model_ecg.h5')
w, h = 144, 224

def read_image(img):
  retrun np.array(Image.open(img).resize((h, w)))[:,:,0:1] / 255

person_1 = read_image(records[0])
person_2 = read_image(records[100])
person_3 = read_image(records[200])
person_3_test = read_image(records[201])

prob = model.predict([person_1.reshape((1, w, h, 1)), person_3_test.reshape((1, w, h, 1))])
pred = (prob>0.5)[0][0]
if pred:
  print("Person verified [confidence: {:.2f}%]".format(100*prob[0][0]))
else: 
  print("Wrong person - [confidence: {:.2f}%]".format(100*(1-prob[0][0])))
