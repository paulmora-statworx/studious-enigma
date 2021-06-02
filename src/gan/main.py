

import numpy as np
import os
import random
from PIL import Image


number_of_small_images = 100
number_range = np.arange(0, 10)

for i in number_range:
    path = f"{os.getcwd()}/data/mnist_large/{i}"
    print(path)
    image_names = os.listdir(path)
    random.shuffle(image_names)
    for image_name in image_names[:number_of_small_images]:
        img_pil = Image.open(f"{os.getcwd()}/data/mnist_large/{i}/{image_name}")
        img_pil.save(f"{os.getcwd()}/data/mnist_small/{i}/{image_name}")
        os.remove(f"{os.getcwd()}/data/mnist_large/{i}/{image_name}")

number_images = 0

for i in number_range:
    path = f"{os.getcwd()}/data/mnist_large/{i}"
    number_images += len(os.listdir(path))




