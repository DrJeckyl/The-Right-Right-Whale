# The-Right-Right-Whale
Right Whale Recognition Kagle Problem


# Directory and Environment Setup
### Step 1:
Download and extract the data from: https://www.kaggle.com/c/noaa-right-whale-recognition/data

* sample_submission.csv
* train.csv
* imgs
* w_7489.jpg
* imgs_subset

### Step 2:
Use the python script inversion.py ("python inversion.py") to sort the images into directories according to their unique whaleID.
(Note: I'm not sure I like this step, but it is required for the next part)

### Step 3:
Run this script to crop some of the labelled data and store it in "/heads" with the same directory structure as the inversion.py created
This command will work on linux:
```
python extract_tiles_from_sloth.py annotations/*.json
```
On Windows (3 separate commands):
```
python extract_tiles_from_sloth.py annotations/whale_faces_jking.json
python extract_tiles_from_sloth.py annotations/whale_faces_smerity.json
python extract_tiles_from_sloth.py annotations/whale_faces_sunil.json
```  
This should set up the directory structure.

I've included a directory called "/negativeInstances" which contains cropped images with no whales in them. They need to be resized to 256x256.
They can be used in conjunction with the images in "/heads" to train the whale face detection model.

# Solution Strategy

The goal here is to classify each image as one of the 447 unique right whales. 
More specifically, we want a vector of probabilities for each image. The probability, p_{ij} corresponds to the probability that image i corresponds to whale j.


To do this there are 2 logical parts. The first, is to create a face detection algorithm that finds the whales in each of the images and crops a box around their heads. (An automated version of what is produced in "/heads")
The second, is to train a classifier on these cropped images on the labelled data in train.csv.

## More detail on Part 1:
I can think of a few ways to accomplish this. The sample submission provided in the competition suggests using a cascade object detector and there are some instructions to do this here: https://www.kaggle.com/c/noaa-right-whale-recognition/details/creating-a-face-detector-for-whales

The .json files in the "/annotation" folder in principle can be converted to .XML files for use in the Matlab code. Word in the forums is that this method is quite crude and miscrops many images.

A perhaps better approach, is to train a model on the images in "/heads" as positive examples and "/negativeInstances" as negative examples. Then, using a sliding window, find the whales in each image. The sliding window will probably need to be done for several different window sizes to account for different zoom factors in the images. Additionally, the "window" will need to be dynamically resized to the size of the training images.

## More detail on Part 2:
This part is fairly open ended, as model selection may play a pretty important role in the performace and there are many different methods to try.

What has been done so far is to brute force try several different algorithms with default settings. From scikit-learn: decision-tree-classifier, Random-Forest and Boosted-Forest, and an SVM have been tried.

From graphlab create, the graphlab.classifier.create method tries several methods and chooses the one with the highest cross validation accuracy. This library includes several implementations of neural nets, so may be useful for extending scikit-learns functionality.

What might be most promising, is that graphlab also includes a pretrained deep learning model from the imagenet database. This is a large collection (1 million plus) images that have had a deep learning neural net trained on, then the features of this data set were extracted at the dropout layer. It's these features from the model which are available for use.

The idea is that graphlab provides a method to extract these features from images which you can use to train a different model on, often getting far better performance than if you had trained a model on the raw images.

# Consistency

For consistency, the images should all be resized to 256x256 jpg images. PIL provides a semi-useful way to do this. Alternatively, graphlab also has a nice tool to quickly resize images.

