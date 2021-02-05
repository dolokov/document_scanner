# import the necessary packages
import numpy as np
import cv2 as cv 
from skimage.filters import threshold_local
import numpy as np
import argparse
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import time 
import json 
import os 
#import imutils

def order_points(pts):
    ## https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    ## https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ 
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    d = np.max([maxWidth, maxHeight])
    maxHeight = int(d * 210./297)
    maxWidth = int(d * 297./210)
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def get_paper_contour(frame):
    optimized = np.array(frame)
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    
    
    ## resize todo
    h, w = frame.shape[:2]
    p = 20

    # convert the image to grayscale, blur it, and find edges
    # in the image
    #optimized = cv.cvtColor(optimized, cv.COLOR_BGR2GRAY)
    blur_kernel = h // 32
    optimized = cv.GaussianBlur(optimized, (blur_kernel, blur_kernel), 0)
    
    ## calculate mean rgb of center patch
    #center_mask = np.zeros(optimized.shape[:],'uint8')
    #center_mask = cv.circle(center_mask, (frame.shape[1]//2,frame.shape[0]//2), frame.shape[0]//8, (1,1,1),3)
    #mean_rgb = np.sum(np.sum(optimized * center_mask,axis=0),axis=0) / (np.pi * frame.shape[0]//8 * frame.shape[0]//8)
    mean_rgb = np.uint8(np.around(np.mean(np.mean(frame[h//2-p:h//2+p,w//2-p:w//2+p,:],axis=0),axis=0)))
    color_pad = 30
    low_rgb = mean_rgb - np.array([color_pad, color_pad, color_pad])
    high_rgb = mean_rgb + np.array([color_pad, color_pad, color_pad])
    mask_rgb = cv.inRange(optimized, low_rgb, high_rgb)
    #print('mean_rgb',mean_rgb,low_rgb,'=>',high_rgb)
    
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts,_ = cv.findContours(mask_rgb.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    paper_cnt = None 
    center = Point(w/2.,h/2.)
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:

            # only accept if center point of screen lies within contour
            if Polygon(approx.reshape((4,2)).tolist()).contains(center):
                paper_cnt = approx
                break
    if 0:
        cv.imshow("Mask (Paper should be white and in center)", mask_rgb)
        cv.waitKey(1)
    return paper_cnt

def get_top_view(frame, cnt, H = 512):
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(frame, np.array(cnt).reshape(4, 2))# * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    #warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    #T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    #warped = (warped > T).astype("uint8") * 255

    warped = np.rot90(warped,2)

    #warped = cv.resize(warped, (int(H * warped.shape[1] / warped.shape[0]), H))

    kernel = np.ones((3,3),np.uint8)
    warped = cv.erode(warped,kernel,iterations = 1)

    return warped 

def main():
    cap = cv.VideoCapture(0)
    paper_data = {} 
    file_paper = os.path.expanduser('~/document_scanner.json')
    if os.path.isfile(file_paper):
        with open(file_paper) as f:
            paper_data = json.load(f)
            if 'cnt' in paper_data:
                paper_data['cnt'] = np.array(paper_data['cnt']).reshape((4,2))
            print('[*] loaded paper points from disk')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # augmented view
        augmented_view = np.array(frame)

        augmented_view = cv.circle(augmented_view, (frame.shape[1]//2,frame.shape[0]//2), frame.shape[0]//4, (0,0,255),3)
        augmented_view = cv.circle(augmented_view, (frame.shape[1]//2,frame.shape[0]//2), frame.shape[0]//32,(0,0,255),3)

        #    paper_cnt = paper_data['cnt']
        #else:
        #if 'cnt' not in paper_data:
        paper_cnt = get_paper_contour(frame)
        #if paper_cnt is not None:
        #    paper_data['cnt'] = paper_cnt
        
        if 'cnt' in paper_data:
            # show the contour (outline) of the piece of paper
            #print("paper_data['cnt']",paper_data['cnt'])
            augmented_view = cv.drawContours(augmented_view, [np.array(paper_data['cnt']).reshape((4,1,2))], -1, (0, 255, 0), 2)
        if paper_cnt is not None:
            # show the contour (outline) of the piece of paper
            augmented_view = cv.drawContours(augmented_view, [paper_cnt], -1, (255, 0, 0), 2)
        
        if 'cnt' in paper_data or paper_cnt is not None:
            if 'cnt' in paper_data:
                top_view = get_top_view(frame, paper_data['cnt'])
            else:
                top_view = get_top_view(frame, paper_cnt)
            top_view_show = cv.resize(top_view,(int(frame.shape[1]*top_view.shape[0]/frame.shape[0]), frame.shape[0]))
            #print('augmented_view', augmented_view.shape,'top_view',top_view.shape,'top_view_show',top_view_show.shape)

            show_view = np.hstack((augmented_view, top_view_show))
            #show_view = top_view
        else:
            show_view = augmented_view

        # Display the resulting frame
        cv.imshow('augmented view', show_view)
        key_pressed = cv.waitKey(1)
        if key_pressed & 0xFF == ord('q'):
            break
        if key_pressed & 0xFF == ord('s'):
            # start countdown if nothing found
            '''if paper_cnt is not None:
                for i in range(3,0,-1):
                    countdown_show = cv.putText(top_view_show, i)
                    show_view_countdown = np.hstack((augmented_view, countdown_show))
                    cv.imshow('augmented view', show_view_countdown)
                    key_pressed = cv.waitKey(1)
                    time.time() - tstart '''
            # save
            if paper_cnt is not None:
                paper_data['cnt'] = paper_cnt
                # save 4 paper points to disk
                with open(file_paper,'w') as f:
                    json_paper_data = {'cnt': paper_cnt.reshape((4,2)).tolist()}
                    f.write(json.dumps(json_paper_data))
                    print('[*] saved extracted points to disk', json_paper_data)

        

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()