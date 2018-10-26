import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from HelperFunctions import draw_lines, weighted_img, getMeans, getLineEndPts

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        # plt.figure()
        # plt.imshow(edges)
        # Next we'll create a masked edges image using cv2.fillPoly()
        right_mask = np.zeros_like(edges)   
        ignore_mask_color = 255   

        # This time we are defining a four sided polygon to mask
        imshape = image.shape
        # print (imshape)

        imageHeight = imshape[0]
        imageWidth = imshape[1]
        upperLeftVertex = imageWidth/2, imageHeight/1.7
        upperRightVertex = imageWidth/2 + imageWidth/34, imageHeight/1.7
        lowerRightVertex = imageWidth*0.925, imageHeight
        lowerLeftVertex = imageWidth/2, imageHeight

        vertices = np.array([[upperLeftVertex, upperRightVertex, lowerRightVertex, lowerLeftVertex]], dtype=np.int32)
        cv2.fillPoly(right_mask, vertices, ignore_mask_color)

        right_masked_edges = cv2.bitwise_and(edges, right_mask)
        right_masked_edges_copy = np.copy(right_masked_edges)*0
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
        lines = cv2.HoughLinesP(right_masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        draw_lines( right_masked_edges_copy, lines )
        # plt.figure()
        # plt.imshow(right_masked_edges_copy)
        
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 30     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 25 # minimum number of pixels making up a line
        max_line_gap = 50    # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(right_masked_edges_copy, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        right_masked_edges_copy = np.copy(right_masked_edges)*0

        draw_lines( right_masked_edges_copy, lines )

        # plt.figure()
        # plt.imshow(right_masked_edges_copy)

        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 80     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50 # minimum number of pixels making up a line
        max_line_gap = 200    # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(right_masked_edges_copy, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
        draw_lines( line_image, lines )

        # plt.figure()
        # plt.imshow(line_image)

        getMeans( line_image, lines )
        # rM, rX, rY = getMeans( lines )
        # rxi, rxf, ryi, ryf = getLineEndPts( image, rM, rX, rY)
        # cv2.line(line_image,(int(rxi),int(ryi)),(int(rxf),int(ryf)),(255,0,0),10)
        # cv2.line(line_image,(int(lxi),int(lyi)),(int(lxf),int(lyf)),(255,0,0),10)

        # draw_lines( line_image, lines )

        # # Iterate over the output "lines" and draw lines on a blank image
        # for line in lines:
        #     for x1,y1,x2,y2 in line:
        #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

        # # Create a "color" binary image to combine with line image
        # color_edges = np.dstack((edges, edges, edges)) 
        # # Draw the lines on the edge image
        # lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

        finalImage = weighted_img(image, line_image)
    
        return finalImage