import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

#Stack images horizontally
def stack_image(scale, arr_img, label = []):
    rows = len(arr_img)
    cols = len(arr_img[0])
    rows_available = isinstance(arr_img[0],list)
    wid = arr_img[0][0].shape[1]
    hei = arr_img[0][0].shape[0]
    #have columns
    if rows_available:
        for x in range(rows):
            for y in range(cols):
                if arr_img[x][y].shape[:2] == arr_img[0][0].shape[:2]:
                    arr_img[x][y] = cv2.resize(arr_img[x][y],(0,0),None,scale,scale)
                else:
                    arr_img[x][y] = cv2.resize(arr_img[x][y],(arr_img[0][0].shape[1],arr_img[0][0].shape[0]),None,scale,scale)
                if len(arr_img[x][y].shape) == 2:
                    arr_img[x][y]= cv2.cvtColor(arr_img[x][y], cv2.COLOR_GRAY2BGR)
        blank_img = np.zeros((hei,wid,3), np.uint8)
        hor = [blank_img]*rows
        hor_con = [blank_img]*rows
        for x in range(rows):
            hor[x] = np.hstack(arr_img[x])
        ver = np.vstack(hor)
    #no column
    else:
        for x in range(rows):
            if arr_img[x].shape[:2] == arr_img[0].shape[:2]:
                arr_img[x] = cv2.resize(arr_img[x], (0, 0), None, scale, scale)
            else:
                arr_img[x] = cv2.resize(arr_img[x], (arr_img[0].shape[1], arr_img[0].shape[0]), None,scale, scale)
            if len(arr_img[x].shape) == 2:
                arr_img[x] = cv2.cvtColor(arr_img[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(arr_img)
        ver = hor

    if len(labels) != 0:
        each_img_w = int(ver.shape[1]/cols)
        each_img_h = int(ver.shape[0]/rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c*each_img_w, each_img_h*d),(c*each_img_w + len(labels[d])*13+27, 30+each_img_h*d),(255,255,255), cv2.FILLED)
                cv2.putText(ver, labels[d], (each_img_w*c+10, each_img_h*d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,255), 2)
    return ver
    

def sift_train_test(train_image, train_gray, test_gray):
    sift = cv2.xfeatures2d.SIFT_create()

    train_keypoints, train_descriptor = sift.detectAndCompute(train_image, None)
    test_keypoints, test_descriptor= sift.detectAndCompute(test_gray, None)

    return train_keypoints, test_keypoints, train_descriptor, test_descriptor



def featureDescriptors(image, method=None):
    assert method is not None, "You need to define a feature detection method. Values are: 'sift',..."

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)
