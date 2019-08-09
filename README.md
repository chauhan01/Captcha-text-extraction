# Captcha-text-extraction


## Introduction
This captcha text extractor uses open cv and keras to extract the text. It uses a CNN model trained on the seprate images of letters of the captcha. 
It takes an input image, output folder, cnn model and a lablebinarizier file and returns an output image with predicted captcha text as image name and also displays the output showing the predicted captcha text.

## How it works

1. Read the image  using open cv
2. Apply threshold
3. Find contours
4. Extract the image regions containing letters
5. Use the trained model to predict the letters
6. Combine the predicted letters
7. Display the results


## Files Description
- captcha.ipynb: It extracts the captcha images into separate images of letters. It also containes the complete project as jupyter notebook. 
- captcha_extractor.py: Extract the captch text, save and display the output captcha image.
- captcha_extractor_model.hdf5: Trained CNN model used for prediction.
- captcha_labels: Holds the labelbinarizer needed to predict the letters.
- test_sample_captcha.png: It is a sample captcha image for testing. 


## Libraries required:
1. open cv
2. keras 


## How to use
captcha_extractor.py needs few arguments:
1. --input: Path to the input png captcha image.
2. --output: path to the output folder.
3. --model: path to the 'captcha_extractor_model.hdf5' file.
4. --labels: Path to the captcha_labels file

## Sample
```
$python captcha_extractor.py --input test__sample_captcha.png --output ./out --model captcha_extractor_model.hdf5 --labels captcha_labels
```
