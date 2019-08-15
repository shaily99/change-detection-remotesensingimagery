# In case you put instance ID into configuration file you can leave this unchanged
INSTANCE_ID = 'e9cf9c52-7c02-476d-90ee-448cfade0826'

import datetime
import numpy as np
from numpy import newaxis
import os
import math
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.misc
from scipy.signal import convolve2d as conv2
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, geo_utils, CustomUrlParam
from skimage import color, data, restoration
from skimage.measure import compare_ssim
import argparse
import imutils
from scipy.misc import imread, imresize, imsave
import ast


def plot_image(imageA, imageB, pts, factor=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # image = cv2.filter2D(image, -1, kernel)

    # astro = color.rgb2gray(image)
    # psf = np.ones((5, 5)) / 25
    # astro = conv2(astro, psf, 'same')
# Add Noise to Image
    # astro_noisy = astro.copy()
    # astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
    # img_bgr = restoration.richardson_lucy(astro_noisy, psf, iterations=30)

    r, g, b = cv2.split(imageA)
    img_bgrA = cv2.merge([b, g, r])
    r, g, b = cv2.split(imageB)
    img_bgrB = cv2.merge([b, g, r])

    # rgb = scipy.misc.toimage(image)

    # height, width = img_bgr.shape
    height, width, depth = img_bgrA.shape
    imgScale = 600/width
    newX, newY = img_bgrA.shape[1]*imgScale, img_bgrA.shape[0]*imgScale
    newimg1 = cv2.resize(img_bgrA, (int(newX), int(newY)))
    newimg2 = cv2.resize(img_bgrB, (int(newX), int(newY)))

    pts = np.array(pts)
# (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped1 = newimg1[y:y+h, x:x+w].copy()
    croped2 = newimg2[y:y+h, x:x+w].copy()

# (2) make mask
    pts = pts - pts.min(axis=0)

    mask1 = np.zeros(croped1.shape[:2], np.uint8)
    cv2.drawContours(mask1, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    mask2 = np.zeros(croped2.shape[:2], np.uint8)
    cv2.drawContours(mask2, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

# (3) do bit-op
    dst1 = cv2.bitwise_and(croped1, croped1, mask=mask1)
    dst2 = cv2.bitwise_and(croped2, croped2, mask=mask2)

# # (4) add the white background
#     bg1 = np.ones_like(croped1, np.uint8)*255
#     cv2.bitwise_not(bg, bg, mask=mask)
#     dst2 = bg + dst

    dst1 = cv2.resize(dst1, (int(newX/2), int(newY/2)))
    #dst1 = cv2.filter2D(dst1, -1, kernel)
    dst2 = cv2.resize(dst2, (int(newX/2), int(newY/2)))
    #   dst2 = cv2.filter2D(dst2, -1, kernel)

    # org, new, diff = find_diff("trialimages", dst1, dst2)
    org, new, diff, score = find_diff("trialimages", dst1, dst2)

    # if np.issubdtype(imageA.dtype, np.floating):
    #     #plt.imshow(np.minimum(image * factor, 1))
    #     # plt.show()
    #     #cv2.imshow('', dst)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     return dst
    # else:
    # plt.imshow(image)
    # plt.show()
    if(score < 1):
        rem = 'yes'
    else:
        rem = 'no'

    ln, br = diff.shape
    ln1, br1, he1 = org.shape
    diff = diff[:, :, newaxis]

    diff = np.tile(diff, (1, 1, 3))
    diff = np.resize(diff, (ln1, br1, he1))
    print (np.shape(diff))
    print (np.shape(org))
    print (np.shape(new))
    numpy_horizontal = np.hstack((org, new))
    numpy_horizontal = np.hstack((numpy_horizontal, diff))

    # cv2.imshow('', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return numpy_horizontal, rem


def deg2rad(degrees):
    return math.pi*degrees/180.0
# radians to degrees


def rad2deg(radians):
    return 180.0*radians/math.pi


# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]


def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt((An*An + Bn*Bn)/(Ad*Ad + Bd*Bd))

# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84


def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/radius
    lonMax = lon + halfSide/radius

    return (rad2deg(latMin), rad2deg(lonMin), rad2deg(latMax), rad2deg(lonMax))


def get_image(id, sp1s, sp1e, sp2s, sp2e):
        # betsiboka_coords_wgs84 = [23.11, 72.71, 23.46, 72.14]
        # betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
    plotid = str(id)
    # print(id)
    # print(type(id))
    file = open('coordinaes.txt', "r")
    temp = [line.strip() for line in file.readlines()]
    file.close()
    lat, lon = 23.02695648, 72.63268738
    bliststr = [[141, 418], [257, 419], [274, 444], [245, 545], [200, 551], [119, 558]]
    for line in temp:
        x = line.split(';')
        if x[0] == plotid:
            lat = float(x[1])
            lon = float(x[2])
            bliststr = x[3]
            break

    bliststr = ast.literal_eval(bliststr)
    print(lat)
    print(lon)
    betsiboka_coords_wgs84 = boundingBox(lon, lat, 2)
    betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)

    wi, he = geo_utils.bbox_to_dimensions(betsiboka_bbox, 2)

    wms_true_color_requestlatest = WmsRequest(layer='TRUE-COLOR-S2-L1C',
                                              bbox=betsiboka_bbox,
                                              # time=('2012-12-01', '2015-12-31'),
                                              time=(sp2s, sp2e),
                                              width=wi, height=he,
                                              maxcc=0.1,
                                              instance_id=INSTANCE_ID)
    wms_true_color_requesthistoric = WmsRequest(layer='TRUE-COLOR-S2-L1C',
                                                bbox=betsiboka_bbox,
                                                time=(sp1s, sp1e),
                                                # time='latest',
                                                width=wi, height=he,
                                                maxcc=0.1,
                                                instance_id=INSTANCE_ID)
    imagenew1 = wms_true_color_requestlatest.get_data()

    imageold1 = wms_true_color_requesthistoric.get_data()

# print(wi)
    # print(he)
    # print('Returned data is of type = %s and length %d.' %
    #       (type(wms_true_color_img), len(wms_true_color_img)))
    #
    # print('Single element in the list is of type {} and has shape {}'.format(type(wms_true_color_img[-1]),
    #                                                                          wms_true_color_img[-1].shape))

    dst, rem = plot_image(imagenew1[-1], imageold1[-1], pts=bliststr)
    return dst, rem, lat, lon


def find_diff(imagepathname, imageA, imageB):
        # load the two input images

    print ("working on -" + imagepathname)

    # imageA = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    # imageA = imresize(imageA, (900,900))
    # # cv2.imshow("trial",imageA)
    # # cv2.waitKey(0)

    # imageB = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    # imageB = imresize(imageB, (900,900))
    # cv2.imshow("trial2",imageB)
    # cv2.waitKey(0)

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    new_size = np.asarray(grayA.shape) / 5
    new_size = new_size.astype(int) * 5
    grayA = imresize(grayA, (new_size)).astype(np.int16)
    grayB = imresize(grayB, (new_size)).astype(np.int16)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    if (score < 1):
        print ('changes detected !')
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

    # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    #     cv2.imshow(imagepathname[:5] + "-O-Marked.jpg", (imageA))
    #     cv2.waitKey(0)
    #     cv2.imshow(imagepathname[:5] + "-N-Marked.jpg", (imageB))
    #     cv2.waitKey(0)
    #     #imsave(imagepath1[:5] + "Diff.jpg", diff)
    #     cv2.imshow(imagepathname[:5] + "-changemap.jpg", thresh)
    #     cv2.waitKey(0)
    #
    # else:
    #     print ('no change')

        return imageA, imageB, thresh, score
