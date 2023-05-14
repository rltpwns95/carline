# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from scipy.optimize import curve_fit
print(cv2.__version__)
# %%

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# %%
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# %%
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# %%
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

# %%
def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# %%
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# %%
def add_imgs(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

# define the true objective function : linear line
def linear(x, a, b):
 return a * x + b

# %%
cap = cv2.VideoCapture("MDR_221009_024928.AVI")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output2.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

ret, frame = cap.read()
rows = frame.shape[0]
cols = frame.shape[1]
img_center = cols/2 - 1#cols/2

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Apply image processing pipeline
        gray = grayscale(frame)
        blur = gaussian_blur(gray, 5)
        edges = canny(blur, low_threshold=50, high_threshold=150)
        imshape = frame.shape
        h_lower = 250
        h_upper = 450
        vertices = np.array([[
            (600-200,rows-h_lower),
            (900-50,rows-h_upper),
            (1100,rows-h_upper),
            (1400, rows-h_lower)
            ]], dtype=np.int32)
        
        vertices2 = np.array(
            [
                [
                    (600,rows-h_lower),
                    (900,rows-h_upper),
                    (950,rows-h_upper),
                    (800,rows-h_lower)
                ],
                [
                    (1100,rows-h_lower),
                    (1050,rows-h_upper),
                    (1080,rows-h_upper),
                    (1250, rows-h_lower)
                ]
            ], dtype=np.int32)
        masked_edges = region_of_interest(edges, vertices2)
        #cv2.line(frame, pt1, pt2, color=[0, 0, 255], thickness=5)
        #cv2.polyines(frame, [vertices], isClosed = True, color=[255, 0, 255], thickness=None, lineType=None, shift=None)
        pts = np.array([[250, 200], [300, 200], [350, 300], [250, 300]])
        cv2.polylines(frame, [vertices], True, (255, 0, 255), 2) # [pts] 리스트 값으로 입력

        lines = hough_lines(masked_edges, 2, np.pi/180, 15, 40, 20)
        #add lines{left, right}, pt{cross}, decision as {left, right, straight}

        line_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        if True:
            line_thresh = 10
            slope_thresh = 0.3
            slope = 0
            slopes = []
            find_lines = []
            right_lines = []
            left_lines = []
            
            #
            # #separateLine
            #
            for line in lines:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                
                dist = math.sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[1]-pt2[1])*(pt1[1]-pt2[1]))
                if dist < line_thresh:
                    continue

                if pt2[0] - pt1[0] == 0:  #코너 일 경우
                    slope = 999.0
                else:
                    slope = (pt2[1] - pt1[1]) /(pt2[0] - pt1[0])
                
                if abs(slope) > slope_thresh :
                    slopes += [slope]
                    find_lines += [(x1, y1, x2, y2)]

            

            for i in range(0, len(find_lines),1) : #find_lines, slopes:
                x1 = find_lines[i][0]
                y1 = find_lines[i][1]
                x2 = find_lines[i][2]
                y2 = find_lines[i][3]
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                
                slope = slopes[i]
                if slope > 0 and pt1[0] > img_center and pt2[0] > img_center : 
                    right_detect = True
                    right_lines += [(x1, y1, x2, y2)]
                    cv2.line(frame, pt1, pt2, color=[0, 0, 255], thickness=5)
                elif slope < 0 and pt1[0] < img_center and pt2[0] < img_center:
                    left_detect = True
                    left_lines += [(x1, y1, x2, y2)]
                    cv2.line(frame, pt1, pt2, color=[255, 0, 0], thickness=5)
            
            dual_lines = [right_lines, left_lines]
            #
            # regression    
            #

            right_points = []
            right_m = 0.0
            right_b = (0,0)
            if right_detect:
                for (x1, y1, x2, y2) in right_lines:
                    pt1 = (x1, y1)
                    pt2 = (x2, y2)
                    right_points+=[(x1,y1)]
                    right_points+=[(x2,y2)]
                    cv2.line(frame, pt1, pt2, color=[255, 255, 0], thickness=5)

                if len(right_points) > 0:
                    #주어진 contour에 최적화된 직선 추출
                    [vx, vy, x, y] = cv2.fitLine(np.array(right_points), cv2.DIST_L2, 0, 0.01, 0.01)
                    right_m = vy/vx #left_line[1] / left_line[0];  //기울기
                    right_b = (x,y) #Point(left_line[2], left_line[3]);

            left_points = []
            left_m = 0.0
            left_b = (0,0)
            if left_detect:
                for (x1, y1, x2, y2) in left_lines:
                    pt1 = (x1, y1)
                    pt2 = (x2, y2)
                    left_points+=[(x1,y1)]
                    left_points+=[(x2,y2)]
                    cv2.line(frame, pt1, pt2, color=[255, 255, 0], thickness=5)

                if len(left_points) > 0:
                    #주어진 contour에 최적화된 직선 추출
                    [vx, vy, x, y] = cv2.fitLine(np.array(left_points), cv2.DIST_L2, 0, 0.01, 0.01)
                    left_m = vy/vx #left_line[1] / left_line[0];  //기울기
                    left_b = (x,y) #Point(left_line[2], left_line[3]);

            #좌우 선 각각의 두 점을 계산한다.
            #y = m*x + b  --> x = (y-b) / m
            y1 = rows - 1
            y2 = rows / 2 - 1   #;// 470;

            if math.fabs(right_m)> 0.0 : 
                right_x1 = ((y1 - right_b[1]) / right_m)+right_b[0]
                right_x2 = ((y2 - right_b[1]) / right_m)+right_b[0]
                pt1 = (int(right_x1[0]), int(y1))
                pt2 = (int(right_x2[0]), int(y2))
                cv2.line(frame, pt1, pt2, color=[255, 255, 0], thickness=5)
            if math.fabs(left_m)> 0.0 : 
                left_x1 = ((y1 - left_b[1]) / left_m)+left_b[0]
                left_x2 = ((y2 - left_b[1]) / left_m)+left_b[0]
                pt1 = (int(left_x1[0]), int(y1))
                pt2 = (int(left_x2[0]), int(y2))
                cv2.line(frame, pt1, pt2, color=[255, 255, 0], thickness=5)


            #
            # draw result
            #
            msg = "Straight"
            threshold = 10
            #두 차선이 교차하는 지점 계산
            x = 0.0
            if left_detect and right_detect:
                x = (((right_m * right_b[0]) - (left_m * left_b[0]) - right_b[1] + left_b[1]) / (right_m - left_m))

                if x >= (img_center - threshold) and x <= (img_center + threshold) :
                    output = "Straight"
                elif x > img_center + threshold :
                    output = "Right Turn"
                elif x < img_center - threshold:
                    output = "Left Turn"
                cv2.putText(frame, msg, (520, 100), cv2.FONT_HERSHEY_PLAIN, 3, color = [255, 255, 255], thickness=3, lineType=cv2.LINE_AA)

            cv2.line(frame, (0, int(rows/2)), (cols-1, int(rows/2)), color = [0, 255, 255], thickness=1)
            cv2.line(frame, (int(cols/2) -1, 0), (int(cols/2)-1, rows), color = [0, 255, 255], thickness=1)


        line_img = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)
        #draw_lines(line_img, lines)

        result = add_imgs(line_img, frame, 0.8, 1., 0.)
        
        

        # Write the processed frame to output video
        out.write(result)
        
        # Display the result
        cv2.imshow('result', result)
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
# %%
