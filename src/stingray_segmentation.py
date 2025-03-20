import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as skmorph
from glob import glob

from myDIP.morphology import fillHoles, removeFragments
from myDIP.evaluation import overallConfusionMatrix, iou


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



if __name__ == "__main__":

    input_dir = r"../Datasets/Stingray_Segmentation/Image"
    groundtruth_dir = r"../Datasets/Stingray_Segmentation/Groundtruth"

    input_path_list = sorted(glob(input_dir + "\*"))
    groundtruth_path_list = sorted(glob(groundtruth_dir + "\*"))

    output_img_list = []
    groundtruth_img_list = []

    for input_path, groundtruth_path in zip(input_path_list, groundtruth_path_list):

        ### -> Read Input Image
        input_img = cv.imread(input_path)
        groundtruth_img = cv.imread(groundtruth_path, 0)
        groundtruth_img = np.where(groundtruth_img > 128, 255, 0)

        ### -> Color model conversion
        hsv_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)

        ### -> Color Range Segmentation
        ray_seg_img = cv.inRange(hsv_img, (3, 70, 0), (18, 200, 255))

        ### -> Morphological Processing
        ray_seg_img = removeFragments(ray_seg_img, 0.0001)  # - Remove small fragments

        stre = skmorph.disk(8)
        ray_seg_img = cv.morphologyEx(ray_seg_img.astype(np.uint8), cv.MORPH_CLOSE, stre) # - Morphological Closing

        ray_seg_img = fillHoles(ray_seg_img) # - Fill Holes

        output_img = getLargestConnectedComponent(ray_seg_img) # - Get Largest Component
        output_img[output_img!=0] = 255

        output_img_list.append(output_img)
        groundtruth_img_list.append(groundtruth_img)

        ### -> Show IoU
        print(f"{os.path.basename(input_path)}: IoU = {iou(output_img, groundtruth_img):.3f}")

        plt.figure()
        plt.imshow(output_img, cmap='gray')

    ### -> Evaluation
    overall_conf_mat = overallConfusionMatrix(output_img_list, groundtruth_img_list)

    ### -> Show Confusion Matrix
    np.set_printoptions(suppress=True, precision=3)
    print(overall_conf_mat)
    plt.show()