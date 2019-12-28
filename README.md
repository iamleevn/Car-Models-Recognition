# Car-Models-Recognition
This model implements car models recognition using tranfer learning with EfficientNetB1 on Stanford Cars dataset.
## Notes
- This projects run on google colab
## Intructions
1. Download pre-trained models from:
[Models](https://drive.google.com/open?id=1h7dEWHWvRSo2NQD6iJHJ92HwyEr8_Cfo)
2. Install Efficient Net
```
!pip install efficientnet
```
3. Load model
```
from keras.models import load_model
import efficientnet.keras as efn

print('Loading model...')
test_model = load_model('my_model0.h5')
test_model.summary()
print('Done!')
```
4. Predict on one images
```
import numpy as np
import matplotlib.pyplot as plt
from efficientnet.keras import center_crop_and_resize, preprocess_input


labels = []
for key in train_generator.class_indices:
    labels.append(key)

# Predict a new image
def predict_one_image(model, filename):
    image = plt.imread(filename)

    # preprocess input
    image_size = 240
    x = center_crop_and_resize(image, image_size=image_size)
    x = np.expand_dims(x, 0)

    # make prediction
    y = model.predict(x)
    
    
    # print top-1 result
    class_id = np.argmax(y)
    res = labels[class_id]
    print("Top 1 result: " + res)
    
        
filename = 'demo/4.jpg'
predict_one_image(test_model1, filename)
image = plt.imread(filename)
plt.imshow(image)
```
## Reference
[Tranfer Learning](http://digital-thinking.de/keras-transfer-learning-for-image-classification-with-effificientnet/)
[Paper](https://arxiv.org/pdf/1905.11946.pdf)
