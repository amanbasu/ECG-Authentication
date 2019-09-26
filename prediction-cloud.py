import os
from googleapiclient import discovery
from PIL import Image
import numpy as np

# set the service-account json key path to your environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path-to-your-service-account-json-key'
# define the model verison hosted on the AI platform
name = 'projects/{}/models/{}/versions/{}'.format('your-project-name', 'your-model-name', 'your-version-name')


def read_img(img):
    return np.array(Image.open(img).resize((224, 144)))[:,:,0:1] / 255

img1 = read_img('path-to-directory/ecg-id-database-filter/Person_01/rec_3_0.png')
img2 = read_img('path-to-directory/ecg-id-database-filter/Person_02/rec_2_0.png')

# create the input request
instances = [{'input_1:0': img1.tolist(), 'input_2:0': img2.tolist()}]
service = discovery.build('ml', 'v1')

# send the request to the model
response = service.projects().predict(
    name=name,
    body={'instances': instances}
).execute()


# receiving the predictions
if 'error' in response:
    raise RuntimeError(response['error'])
else:
    pred = response['predictions'][0]['dense_1/Sigmoid:0'][0]
    print("Similarity between two images: {:.2f}%".format(pred*100))
    