# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
# Read in and grayscale the image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from HelperFunctions import weighted_img, slopeSorter, draw_lines, extrapolateLine, getMeans, getLineEndPts

plt.ion()

image_list = os.listdir("../test_images/")


for imageTitle in image_list:
	image = mpimg.imread('../test_images/' + imageTitle)
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)   
	ignore_mask_color = 255   

	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	# print (imshape)

	imageHeight = imshape[0]
	imageWidth = imshape[1]
	upperLeftVertex = imageWidth/2 - imageWidth/34, imageHeight/1.7
	upperRightVertex = imageWidth/2 + imageWidth/34, imageHeight/1.7
	lowerRightVertex = imageWidth*0.925, imageHeight
	lowerLeftVertex = imageWidth*0.075, imageHeight

	# vertices = np.array([[(540,150), (250,420), (250,460), (540,850)]], dtype=np.int32)
	vertices = np.array([[upperLeftVertex, upperRightVertex, lowerRightVertex, lowerLeftVertex]], dtype=np.int32)
	# vertices = np.array([[(imshape[1]/2 - imshape[1]/32, imshape[0]/2 + imshape[0]/19), (850,imshape[0]/2 + imshape[0]/19), (0,imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
	# print (vertices)
	#vertices = np.array([[(400,imshape[1]/2),(200, imshape[1]/2), (0,imshape[1]), (imshape[1],imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	masked_edges = cv2.bitwise_and(edges, mask)
	masked_edges_copy = np.copy(masked_edges)*0

	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 10     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 8 # minimum number of pixels making up a line
	max_line_gap = 10    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	# rM, rX, rY, lM, lX, lY = getMeans( lines )
	# rxi, rxf, ryi, ryf, lxi, lxf, lyi, lyf = getLineEndPts( image, rM, rX, rY, lM, lX, lY )

	draw_lines( masked_edges_copy, lines )

	rho = 1
	theta = np.pi/180
	threshold = 10
	min_line_length = 20
	max_line_gap = 250
	lines = cv2.HoughLinesP(masked_edges_copy, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

	draw_lines( line_image, lines )

	# rho = 1
	# theta = np.pi/180
	# threshold = 70
	# min_line_length = 40
	# max_line_gap = 300
	# lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
 #                                min_line_length, max_line_gap)

	# xiR, xfR, yiR, yfR, xiL, xfL, yiL, yfL = getLineEndPts( image, lines )

	# cv2.line(image,(rxi,ryi),(rxf,ryf),(255,0,0),10)
	# cv2.line(image,(xiL,yiL),(xfL,yfL),(255,0,0),10)

	# plt.figure()
	# plt.imshow(masked_edges)
	# plt.savefig('../test_images_output/' + "maskedEdges.jpg")
	

	# Iterate over the output "lines" and draw lines on a blank image
	# prev_line = False
	# for line in lines:
	# 	# if( prev_line = True ):

	#     for x1,y1,x2,y2 in line:
	#         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

	# # Create a "color" binary image to combine with line image
	# color_edges = np.dstack((edges, edges, edges)) 
	# # Draw the lines on the edge image
	# lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

	finalImage = weighted_img(image, line_image)
	# finalImage = masked_edges_copy

	plt.figure()
	plt.imshow(finalImage)
	plt.savefig('../test_images_output/' + imageTitle)




