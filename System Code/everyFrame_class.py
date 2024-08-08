import cv2
import numpy as np

class EveryFrame:
  def __init__(self, image_width, image_height):
    self.IMAGE_WIDTH = image_width
    self.IMAGE_HEIGHT = image_height

  def frame_preprocessing(self, image):
    image = image[:, :, [2, 1, 0]]  # convert BGR to RGB
    image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)) # resizing image
    image = image.astype(np.float32) # convert image pixel to float32
    image = image / 255.0 # normalization
    return image

  def model_predict(self, frames, model):
    get_frame = [self.frame_preprocessing(image) for image in frames] # preprocessing video
    get_frame = np.array(get_frame)
    get_frame = np.expand_dims(get_frame, axis=0)
    return model.predict(get_frame) # predict frame

  def show_predict(self, predict):
    label = np.argmax(predict) # get maks indeks value
    if label == 0:
      text = 'Normal'
    elif label == 1:
      text = 'Pencurian'
    else:
      text = 'Detecting...'
    return text