# change-detection-remotesensingimagery
This repository contains the project "Change Detection in Land Usage using Remote Sensing Imagery" completed at Bhaskaracharya Insitutute for Space Applications and Geo-Informatics under Dr. Manoj Pandya from May 2019 to July 2019

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

#### For Detailed methodology, refer to the project report.
