import cv2
import numpy as np
import copy
from PIL import Image, ImageEnhance
import os.path

def SAP(src, percentage):
    img = src.copy()
    num = int(percentage*src.shape[0]*src.shape[1])
    for i in range(num):
        R = np.random.randint(0, src.shape[0]-1)
        G = np.random.randint(0, src.shape[1]-1)
        B = np.random.randint(0, 3)
        if np.random.randint(0,1) == 0:
            img[R, G, B] = 0
        else:
            img[R, G, B] = 255
    return img

def GN(src, percentage):
    img = src.copy()
    w = img.shape[1]
    h = img.shape[0]
    num = int(percentage*w*h)
    for i in range(num):
        tx = np.random.randint(0, h)
        ty = np.random.randint(0, w)
        img[tx][ty][np.random.randint(3)] = np.random.randn(1)[0]
    return img

def darker(src, percentage=0.9):
    img = src.copy()
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            img[j, i, 0] = int(img[j, i, 0]*percentage)
            img[j, i, 1] = int(img[j, i, 1]*percentage)
            img[j, i, 2] = int(img[j, i, 2]*percentage)
    return img

def brighter(src, percentage=1.5):
    img = src.copy()
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            img[j, i, 0] = np.clip(int(img[j, i, 0]*percentage), a_max=255, a_min=0)
            img[j, i, 1] = np.clip(int(img[j, i, 1]*percentage), a_max=255, a_min=0)
            img[j, i, 2] = np.clip(int(img[j, i, 2]*percentage), a_max=255, a_min=0)
    return img

def colour(src):
    img = src.copy()
    # saturation = np.random.randint(0, 1)
    # contrast = np.random.randint(0, 1)
    # sharpness = np.random.randint(0, 1)
    # if np.random.random() < saturation:
    rf = np.random.randint(0, 31) / 10.
    img = ImageEnhance.Color(img).enhance(rf)
    # if np.random.random() < contrast:
    rf = np.random.randint(10, 21) / 10.
    img = ImageEnhance.Contrast(img).enhance(rf)
    # if np.random.random() < sharpness:
    rf = np.random.randint(0, 31) / 10.
    img = ImageEnhance.Sharpness(img).enhance(rf)
    return img


def rotate(src, angle, center=None, scale=1.0):
    h, w = src.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(src, m, (w, h))
    return rotated

def flip(src):
    img =np.fliplr(src)
    return img


root = '/home/deep211/v/img/train/'
for file in os.listdir(root):
    ipath = root + file
    img = Image.open(ipath)
    img = img.resize((224, 224), Image.BICUBIC)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    rotated_90 = rotate(img, 90)
    rotated_180 = rotate(img, 180)
    cv2.imwrite(root + file[0:-4] + '__90.jpg', rotated_90)
    cv2.imwrite(root + file[0:-4] + '__180.jpg', rotated_180)

for file in os.listdir(root):
    img = cv2.imread(root + file)
    fimg = flip(img)
    cv2.imwrite(root + file[0:-4] + '__flip.jpg', fimg)

    gimg = GN(img, 0.3)
    cv2.imwrite(root + file[0:-4] + '__GN.jpg', gimg)

    dimg = darker(img)
    cv2.imwrite(root + file[0:-4] + '__dark.jpg', dimg)

    limg = brighter(img)
    cv2.imwrite(root + file[0:-4] + '__light.jpg', limg)

    blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    cv2.imwrite(root + file[0:-4] + '__blur.jpg', blur)

    for i in range(0, 3):
        cimg = colour(img)
        cv2.imwrite(root + file[0:-4] + f'__c{i}.jpg', cimg)




