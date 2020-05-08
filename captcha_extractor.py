import pandas as pd
import numpy as np
import cv2
import imutils
from imutils import paths
import os
import os.path
import pickle
import matplotlib.pyplot as plt
from IPython.display import Image, display

from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the captcha png image")
ap.add_argument("-o","--output", required=True, help="path to save the output image")
ap.add_argument("-m", "--model", required=True, help="path to the hdf5 model")
ap.add_argument("-lb", "--labels", required=True, help="path to the captcha labels dat file")
args = vars(ap.parse_args())


#Loading the model
model = load_model(args['model'])
with open(args['labels'], "rb") as f:
    lb = pickle.load(f)

# Load the image and convert it to grayscale
image = cv2.imread(args['input'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# Adding some extra padding around the image
gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

# applying threshold
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

# find the contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    
letter_image_regions = []

# Now we can loop through each of the contours and extract the letter

for contour in contours:
    # Get the rectangle that contains the contour
    (x, y, w, h) = cv2.boundingRect(contour)
    
    # checking if any counter is too wide
    # if countour is too wide then there could be two letters joined together or are very close to each other
    if w / h > 1.25:
        # Split it in half into two letter regions
        half_width = int(w / 2)
        letter_image_regions.append((x, y, half_width, h))
        letter_image_regions.append((x + half_width, y, half_width, h))
    else:
        letter_image_regions.append((x, y, w, h))
            

# Sort the detected letter images based on the x coordinate to make sure
# we get them from left-to-right so that we match the right image with the right letter  

letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

# Create an output image and a list to hold our predicted letters
output = cv2.merge([gray] * 3)
predictions = []
    
# Creating an empty list for storing predicted letters
predictions = []
    
# Save out each letter as a single image
for letter_bounding_box in letter_image_regions:
    # Grab the coordinates of the letter in the image
    x, y, w, h = letter_bounding_box

    # Extract the letter from the original image with a 2-pixel margin around the edge
    letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

    letter_image = cv2.resize(letter_image, (30,30))
        
    # Turn the single image into a 4d list of images
    letter_image = np.expand_dims(letter_image, axis=2)
    letter_image = np.expand_dims(letter_image, axis=0)

    # making prediction
    pred = model.predict(letter_image)
        
    # Convert the one-hot-encoded prediction back to a normal letter
    letter = lb.inverse_transform(pred)[0]
    predictions.append(letter)


    # draw the prediction on the output image
    cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
    cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)



# Print the captcha's text
captcha_text = "".join(predictions)
print("CAPTCHA text is: {}".format(captcha_text))

# Get the folder to save the image in
save_path = os.path.join(args['output'], captcha_text)

p = os.path.join(save_path+'.png' )
#writing the image to the output folder
cv2.imwrite(p, image)
cv2.imwrite("output.png",output)
print("Output saved to "+args['output'])
# Show the annotated image
# cv2.imshow("Output", output)
# cv2.waitKey()

