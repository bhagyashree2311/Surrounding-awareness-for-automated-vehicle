# import the necessary packages
from object_detector import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
bboxes = []
imagePaths = []
for row in open(config.ANNOTS_PATH+"/annotations.csv").read().strip().split("\n")[1:]:
    (_,image_name,width,height,startX,startY,endX,endY) = row.split(",")

    imagePath = os.path.sep.join([config.IMAGES_PATH,image_name])
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    y_ = image.shape[0]
    x_ = image.shape[1]
    image = cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)

    targetSize = 224
    x_scale = targetSize / x_
    y_scale = targetSize / y_

    startX = int(np.round(float(startX) * x_scale))
    startY = int(np.round(float(startY) * y_scale))
    endX = int(np.round(float(endX) * x_scale))
    endY = int(np.round(float(endY) * y_scale))


    startX_ = float(startX)/float(targetSize)
    startY_ = float(startY)/float(targetSize)

    endX_ = float(endX)/float(targetSize)
    endY_ = float(endY)/float(targetSize)



    image = img_to_array(image)

    data.append(image)
    bboxes.append((startX_,startY_,endX_,endY_))
    imagePaths.append(imagePath)

# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range
# [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)



split = train_test_split(data, bboxes, imagePaths,test_size=0.10, random_state=9)
del data
del bboxes
del imagePaths
# unpack the data split
(trainImages, valImages) = split[:2]
(trainBBoxes, valBBoxes) = split[2:4]
(trainPaths, valPaths) = split[4:]

del split
# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(valPaths))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights=None, include_top=False,input_shape=(224, 224, 3))

vgg.trainable = True
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label

# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
	inputs=vgg.input,
	outputs=bboxHead)


# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

# train the network for bounding box regression
print("[INFO] training model...")
H = model.fit(
	trainImages, trainBBoxes,
	validation_data=(valImages, valBBoxes),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# plot the total loss, label loss, and bounding box loss
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOTS_PATH)
