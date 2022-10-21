import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np

NAMES = {0: 'apple',
        1: 'airplane',
        2: 'laptop',
        3: 'banana',
        4: 'star',
        5: 'rainbow',
        6: 'TheEiffelTower',
        7: 'bridge'}

model = load_model("doodle_model_v1")

_, D = model.layers[0].input_shape
D = int(np.sqrt(D))

def getBoxCoords(image):
    cy, cx, _ = np.array(image.shape) // 2
    p = image.shape[1] // 5

    r1 = (cx - p//2, cy - p//2)
    r2 = (cx + p//2, cy + p//2)

    return r1, r2

def getBox(image):
    r1, r2 = getBoxCoords(image)
    cropped = image[r1[1]:r2[1], r1[0]:r2[0]]

    return cropped

def getInputImage(image):
    box = getBox(image)

    gray = cv.cvtColor(box, cv.COLOR_BGR2GRAY)

    resized = cv.resize(gray, (D, D), interpolation = cv.INTER_AREA)
    resized_avg = np.sum(resized) / (D*D)
    resized_max = np.max(resized)

    resized_tresh = cv.threshold(resized, resized_max*0.9, 255, cv.THRESH_BINARY)[1]

    input_image = resized_tresh.flatten()
    input_image = 1 - np.array([input_image]) / 255

    return resized_tresh, input_image



cam = cv.VideoCapture(0)

while True:
    ret, image = cam.read()

    resized_tresh, input_image = getInputImage(image)

    p = model.predict(input_image)

    if max(p[0]) > 0:
        name = NAMES[np.argmax(p)]
        print("Doodle =", name)

    r1, r2 = getBoxCoords(image)

    cv.rectangle(image, r1, r2, (255,0,0), 2)

    resized = cv.resize(resized_tresh, (D*3, D*3), interpolation = cv.INTER_AREA)
    resized_bgr = cv.cvtColor(resized, cv.COLOR_GRAY2BGR)
    
    image[r2[1]-D*3:r2[1], r2[0]:r2[0]+D*3] = resized_bgr

    cv.imshow('resized', resized)

    cv.imshow("Doodle", image)
    if cv.waitKey(1) == ord('q'): 
        break


cam.release()

cv.destroyAllWindows()