import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1) > 20:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def slopeSorter( initialList ):
    leftLane = np.array([[[]]], dtype=np.int32)
    rightLane = np.array([[[]]], dtype=np.int32)
    for line in initialList:
        for x1,y1,x2,y2 in line:
            if (y2-y1)/(x2-x1) > 0:
                rightLane = np.append(rightLane, line, axis=0)
            else:
                leftLane = np.append(leftLane, line, axis=0)
                
    return leftLane, rightLane

def extrapolateLine( lineList ):
    xiNew = 0
    yiNew = 0
    firstIteration = True
    for line in lineList:
        for x1,y1,x2,y2 in line:
            if firstIteration == False:
                newLine = xiNew,yiNew,x1,y1
                lineList = np.append(lineList, newLine, axis=1)
            xiNew = x2
            yiNew = y2
            firstIteration = False

    return lineList

def getMeans( lineList ):
    rightSlopeAggregate = 0
    leftSlopeAggregate = 0
    rightX = 0
    leftX = 0
    rightY = 0
    leftY = 0
    rCounter = 0
    lCounter = 0
    for line in lineList:
        for x1,y1,x2,y2 in line:
            if x1 != x2 and (y2-y1)/(x2-x1) > 0 and abs((x1+x2)/2) > 960/2:
                rightSlopeAggregate += (y2-y1)/(x2-x1)
                rightX += x1 + (x2-x1)/2
                rightY += y1 + (y2-y1)/2
                rCounter += 1
            elif x1 != x2 and (y2-y1)/(x2-x1) < 0 and abs((x1+x2)/2) < 960/2:
                slope = (y2-y1)/(x2-x1)
                leftSlopeAggregate += slope
                leftX += x1 + (x2-x1)/2
                leftY += y1 + (y2-y1)/2
                lCounter += 1

    rightSlope = rightSlopeAggregate/rCounter
    rX = rightX/rCounter
    rY = rightY/rCounter
    leftSlope = leftSlopeAggregate/lCounter
    lX = leftX/lCounter
    lY = leftY/lCounter
            
    return rightSlope, rX, rY, leftSlope, lX, lY

def getLineEndPts( image, lineList ):
    imshape = image.shape
    imHeight = imshape[0]
    imWidth = imshape[1]
    xfRight = imWidth
    xiRight = 0
    yiRight = imHeight
    yfRight = imHeight
    xfLeft = 0
    xiLeft = imWidth
    yiLeft = imHeight
    yfLeft = imHeight
    for line in lineList:
        for x1,y1,x2,y2 in line:
            if x1 != x2 and (y2-y1)/(x2-x1) > 0 and abs((x1+x2)/2) > imWidth/2:
                if x1 < xfRight:
                    xfRight = x1
                if x2 > xiRight:
                    xiRight = x2
                if y1 < yfRight:
                    yfRight = y1
            elif x1 != x2 and (y2-y1)/(x2-x1) < 0 and abs((x1+x2)/2) < imWidth/2:
                if x2 > xfLeft:
                    xfLeft = x2
                if x1 < xiLeft:
                    xiLeft = x1
                if y1 < yfLeft:
                    yfLeft = y1
    return int(xiRight), int(xfRight), int(yiRight), int(yfRight), int(xiLeft), int(xfLeft), int(yiLeft), int(yfLeft)

def getLineEndPts( image, rM, rX, rY, lM, lX, lY ):
    imshape = image.shape
    imHeight = imshape[0]
    imWidth = imshape[1]
    rxi = (imHeight - rY)/rM + rX
    rxf = rxi - 2*(rxi - rX)
    ryi = imHeight
    ryf = imHeight - 2*(imHeight - rY)
    lxi = (imHeight - lY)/lM + lX
    lxf = lxi - 2*(lxi - lX)
    lyi = imHeight
    lyf = imHeight - 2*(imHeight - lY)

    return rxi, rxf, ryi, ryf, lxi, lxf, lyi, lyf