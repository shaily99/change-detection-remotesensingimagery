# In case you put instance ID into configuration file you can leave this unchanged
INSTANCE_ID = 'e9cf9c52-7c02-476d-90ee-448cfade0826'

import datetime
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
import cv2
import scipy.misc
from scipy.signal import convolve2d as conv2
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, geo_utils
from skimage import color, data, restoration


    
def plot_image(image, factor=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    #image = cv2.filter2D(image, -1, kernel)
    
    
    #astro = color.rgb2gray(image)
    #psf = np.ones((5, 5)) / 25
    #astro = conv2(astro, psf, 'same')
# Add Noise to Image
    #astro_noisy = astro.copy()
    #astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
    #img_bgr = restoration.richardson_lucy(astro_noisy, psf, iterations=30)

   
    r, g, b = cv2.split(image)
    img_bgr = cv2.merge([b, g, r])
    #rgb = scipy.misc.toimage(image)
    
    #height, width = img_bgr.shape
    height, width, depth = img_bgr.shape
    imgScale = 600/width
    newX,newY = img_bgr.shape[1]*imgScale, img_bgr.shape[0]*imgScale
    newimg = cv2.resize(img_bgr,(int(newX),int(newY)))
   
    pts = np.array([[141,418],[257,419],[274,444],[245,545],[200,551],[119,558]])

## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = newimg[y:y+h, x:x+w].copy()

## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
   
    
   
    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
        plt.show()
        #cv2.imshow('', dst)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        #plt.imshow(image)
        #plt.show()
        dst = cv2.resize(dst,(int(newX/2),int(newY/2)))
        dst = cv2.filter2D(dst, -1, kernel)
        cv2.imshow('', newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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


if __name__ == "__main__":
        #betsiboka_coords_wgs84 = [23.11, 72.71, 23.46, 72.14]
        #betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
    betsiboka_coords_wgs84 = boundingBox(72.530434,23.076670, 2)
    betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)

    wi, he = geo_utils.bbox_to_dimensions(betsiboka_bbox, 2)

    wms_true_color_request = WmsRequest(layer='TRUE-COLOR-S2-L1C',
                                        bbox=betsiboka_bbox,
                                        time=('2015-01-01', '2016-12-31'),
                                        width=wi, height=he,
                                        maxcc=0.1,
                                        instance_id=INSTANCE_ID)

    wms_true_color_img = wms_true_color_request.get_data()
    print(wi)
    print(he)
    print('Returned data is of type = %s and length %d.' %
          (type(wms_true_color_img), len(wms_true_color_img)))

    print('Single element in the list is of type {} and has shape {}'.format(type(wms_true_color_img[-1]),
                                                                             wms_true_color_img[-1].shape))

    plot_image(wms_true_color_img[-1])
