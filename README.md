# change-detection-remotesensingimagery
This repository contains the project "Change Detection in Land Usage using Remote Sensing Imagery" completed at Bhaskaracharya Insitutute for Space Applications and Geo-Informatics under Dr. Manoj Pandya from May 2019 to July 2019 by me and my project partner Meet Kanani.

### About BISAG
Modern day planning for inclusive development and growth calls for transparent,efficient, effective,responsive and low cost decision making systems involving multi-disciplinary information such that it not only encourages people's participation, ensuring equitable development but also takes into account the sustainability of natural resources. The applications of space technology and Geo–informatics have contributed significantly towards the socio-economic development. Taking cognizance of the need of geo-spatial information for developmental planning and management of resources, the Department of Science and Technology, **Government of Gujarat established "Bhaskaracharya Institute for Space Applications and Geo-informatics" (BISAG). BISAG is an ISO 9001:2008, ISO 27001:2005 and CMMI: 5 certified institute.** BISAG which was initially set up to carryout space technology applications, has evolved into a **centre of excellence, where research and innovations are combined with the requirements of users and thus acts as a value added service provider, a technology developer and as a facilitator for providing direct benefits of space technologies to the grass root level functions/functionaries.**

## The Problem

The problem of illegal constructions on land plots is of grave concern for the government and citizens alike. The governemnt has enhanced presnt system for thwarting illegal construction activities by including information obtained from satellite imagery to monitor flagged lans of plots. However, the presntly established system, includes manual mapping of a large number of plots and relies on detection of changes from the multi-temporal satellite manually reported by operators at BISAG. This makes the task not only cumbersome, but also limits the scalability.

## Project Objectives:

Our objective was to develop the prototype of a computational technique that would reduce the man-power required for the task by automating the process of change detection. Further the technique was to be rendered to the end-user within an easy to operate web application.

## Flow of the Project:

1. Obtain required data from sentinel hub for the given coordinates at the specified dates.
2. Process the images using OpenCV to detect changes between the two obtained images.
3. Create a user friendly application that allows the user to select the plot of interest and duration of monitoring.
4. Display change maps and labeled change images if changes are detected and send an alert to the central database.

## Tools Used
1. **Python 3**: A common purpose programming language
2. **OpenCV**: A computer vision library for python
3. **Sentinel Hub**: A cloud based GIS platform for distirbution, management and analysis of staellite data provided by open data sources (The Copernicus Open Access Hub) that provides complete, free and open access to Sentinei - 1,2,3 and 5P. The python package of Sentinel Hub allows users to make OGC(WMS and WCS) web requests to download and process satellite images within python scripts.
4. **Flask**: A python based micrframework used to develop a prototype web application for end user.

## Methodology
1. Two images for the specified time and coordinates is obtained from request to sentinelhub.
2. The image is cropped and sharpened to focus area of interest.
3. Structural Similarity Index (SSIM) is used to detect changes.
4. If changes are detected, change maps are displayed.

#### For Detailed methodology, refer to the [project report]( change-detection-remotesensingimagery/Change-detection-in-land-usage-using-remote-sensing-imagery.pdf )

## Using the project:
1. The coordinates along with area of interests should be added to the coordinates.txt file. This file contains the id, coordinates and bounding box for area of interest
2. Run python3 getextent.py with the coordinates of interest and obtain the bounding box for interest in the displayed image. Now copy these local coordinates to the coordinates.txt file
3. Run the flask application by using pyhton3 flk.py
4. Once the interface is displayed, enter the id of the corrdinates that you wish to process along with the duration of monitoring. Note You can enter only valid ids.
5. The results are displayed on the screen, which include the change maps and labelled changes. In the backend an api request sends an alert to the central database in the event that changes are detected.

## References:
1. Adrian Rosebroke, Comparison of images using SSIM, September 15, 2014 retrieved
from ​ https://www.pyimagesearch.com/2014/09/15/python-compare-two-images

2. Documentation on SentinelHub python package retrieved from
https://sentinelhub-py.readthedocs.io/en/latest/

3. Flask Documentation retrieved from ​ http://flask.pocoo.org/docs/1.0/tutorial/

4. Iftekher Mamun, Image classification using SSIM, Jan 2017 retrieved from
https://towardsdatascience.com/image-classification-using-ssim-34e549ec6e12

5. OpenCV Documentation and tutorials retrieved from
https://docs.opencv.org/2.4/doc/tutorials/tutorials.html

6. Turgay Celik, “Unsupervised change detection in satellite images using Principal
Component Analysis and K-means clustering”, IEEE Geoscience and Remote Sensing
Letters, Vol. 6, No.4, October 2009.

7. Unsupervised Change Detection in Multi-Temporal Satellite Images using PCA &
K-Means : Python code, retrieved from
https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-mult
i-temporal-satellite-images-using-pca-k-means-python-code
