from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import os

model = load_model(os.path.join("checkpoint", "inceptionV3.137-0.999-0.999-0.999-1.000-0.998.hdf5"))

 # 加载图像
img = load_img(img_path, target_size=(299, 299))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)
answers = []
for p in predictions:
  p=np.array(p[0])
  answers.append(p.argmax())
print(answers)
