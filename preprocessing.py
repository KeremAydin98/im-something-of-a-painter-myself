import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12,12))

monet_img = cv2.imread("./monet_jpg/0a5075d42a.jpg")
monet_img = cv2.cvtColor(monet_img, cv2.COLOR_BGR2RGB)

photo_img = cv2.imread("./photo_jpg/0a0c3a6d07.jpg")
photo_img = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)

ax[0].imshow(monet_img / 255)
ax[1].imshow(photo_img / 255)

plt.show()

def preprocess_image(image):

    image = cv2.image.resize(image, (256, 256))

    image = np.array(image)

    image = image / 255

    return image

def load_images(source_paths, target_paths):

    source_images = []
    target_images = []

    for target_path, source_path in zip(os.listdir(target_paths), os.listdir(source_paths)):

        source_img = cv2.imread(os.path.join(source_paths, source_path))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

        source_img = preprocess_image(source_img)

        source_images.append(source_img)

        target_img = cv2.imread(os.path.join(target_paths, target_path))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        target_img = preprocess_image(target_img)

        target_images.append(target_img)

    
    return np.array(source_images), np.array(target_images)

