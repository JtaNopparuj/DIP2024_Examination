import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
from myDIP.morphology import fillHoles
from skimage.measure import regionprops
from myDIP.evaluation import overallConfusionMatrix, confusionMatrix
import skimage.morphology as skmorph

def getLargestConnectedComponent(binary_img):
    '''
        Get the largest component in the Input Image
    '''
    binary_img = binary_img.astype(np.uint8)
    # -> Connected Components
    _, label_img = cv.connectedComponents(binary_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    # -> Remove Background
    counts = counts[labels!=0]
    labels = labels[labels!=0]

    # -> Get largest component
    largest_group_label = labels[np.argmax(counts)]
    output_img = np.zeros_like(binary_img, np.uint8)
    output_img[label_img==largest_group_label] = 1

    return output_img

def segmentation(input_img):
    '''
        Date Fruit Segmentation
    '''
    thres, seg_img = cv.threshold(input_img, None, 1, cv.THRESH_BINARY_INV+cv.THRESH_OTSU) # - Otsu Thresholding
    seg_img = fillHoles(seg_img) # - Fill Holes
    seg_img = getLargestConnectedComponent(seg_img) # - Get Largest Component

    return seg_img


def featureExtraction(seg_img, input_img=None):
    '''
        Date Fruit Feature Extraction
    '''
    _, label_img = cv.connectedComponents(seg_img)
    object_list = regionprops(label_img, intensity_image=input_img)

    if input_img is not None:
        # -> Convert to HSV Color Model
        hsv_img = cv.cvtColor(input_img.astype(np.uint8), cv.COLOR_RGB2HSV)

    for object in object_list:

        y_indices, x_indices = np.where(label_img == object.label)
        # -> Find color mean (HSV)
        h_mean, s_mean, v_mean = np.mean(hsv_img[y_indices, x_indices], axis=0)
        # -> Area
        area = object.area

    # -> Feature Vector
    feature_vector = [h_mean, area]
    
    return feature_vector

def classifier(feature_vector):
    '''
        Manual Decision Tree Classifier
    '''
    hue_mean, area = feature_vector

    if area >= 60000:
        pred_class = "Medjool"
    elif area >= 40000 and area < 60000:
        pred_class = "Rutab"
    else:
        if hue_mean < 10:
            pred_class = "Galaxy"
        else:
            pred_class = "Ajwa"

    return pred_class


if __name__ == "__main__":

    ### -> Set base directory
    base_dir = r"../Datasets/Date_Fruit_Classification"

    ### -> Get class name from folder name
    class_name = os.listdir(base_dir)

    y_true = []
    y_pred = []

    for class_label in class_name:

        ### -> Class Directory
        class_dir = os.path.join(base_dir, class_label)
        img_path_list = sorted(glob(class_dir + "\*")) # - Get Image PATH

        for img_path in img_path_list:

            ### -> Read Input Image
            img = cv.imread(img_path)

            ### -> Color Model Conversion
            rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ### -> Segmentation
            seg_img = segmentation(gray_img)

            ### -> Feature Extraction
            feature_vector = featureExtraction(seg_img, rgb_img)

            ### -> Classification
            pred_class = classifier(feature_vector)

            y_true.append(class_label)
            y_pred.append(pred_class)

    ### -> Evaluation
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    conf_mat = confusionMatrix(y_pred, y_true)

    ### -> Show Confusion Matrix
    np.set_printoptions(suppress=True, precision=3)
    print(conf_mat)