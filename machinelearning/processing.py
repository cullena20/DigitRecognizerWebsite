from PIL import Image
import io
import base64
import numpy as np

def base64_to_PIL(b64):
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    return image

def show_image(image):
    image.show()

def process_image(image, resize=False):
    image = image.resize((28, 28), resample=Image.BILINEAR)
    image = np.array(image)
    image = np.ones((28, 28)) * 255 - image
    image = image / 255
    if resize:
        image = shrink_image(image)
        image = resize_image(image, (20, 20))
        image = add_padding(image, (28, 28))
    image = np.reshape(image, (1, 28, 28))
    return image

def partially_process(image, resize=False):
    image = image.resize((28, 28), resample=Image.BILINEAR)
    image = np.array(image)
    image = np.ones((28, 28)) * 255 - image
    if resize:
        image = shrink_image(image)
        image = resize_image(image, (20, 20))
        image = add_padding(image, (28, 28))
    image = Image.fromarray(image)
    return image

# get rid of black rows and columns
def shrink_image(img):
    while np.sum(img[0]) == 0:
        img = img[1:]
    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)
    while np.sum(img[-1]) == 0:
        img = img[:-1]
    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)
    return img

# resize image to fit into target size, here it will be used to turn shrunk image into 20x20 image
def resize_image(image, target_size):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(target_size)
    resized_array = np.array(resized_image)

    return resized_array

# add padding to make image fit into target size, here it will be used to turn 20x20 images into 28x28 images
def add_padding(image, target_size):
    rows, cols = image.shape
    target_rows, target_cols = target_size
    
    pad_row = (target_rows - rows) // 2
    pad_col = (target_cols - cols) // 2
    
    padded_image = np.zeros(target_size)
    padded_image[pad_row:pad_row+rows, pad_col:pad_col+cols] = image
    
    return padded_image