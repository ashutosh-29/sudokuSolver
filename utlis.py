
import cv2
import numpy as np
from tensorflow.keras.models import load_model

################################### Pre-processing and final image generation ###############################

def preProcess(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(7,7),3)
    imgThreshold=cv2.adaptiveThreshold(imgBlur,255,1,1,11,2)
    lines=cv2.HoughLinesP(imgThreshold,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(imgThreshold,(x1,y1),(x2,y2),(255,0,0),2)
    return imgThreshold
def biggestContours(contour):
    biggest=np.array([])
    maxarea=0
    for i in contour:
        area=cv2.contourArea(i)
        if area>10000:
            perimeter=cv2.arcLength(i,True)
            aprx=cv2.approxPolyDP(i,0.02*perimeter,True)
            if area>maxarea and len(aprx)==4:
                biggest=aprx
                maxarea=area
                print(maxarea)
    return maxarea,biggest
def reorder(points):
    points=points.reshape((4,2))
    newPoints=np.zeros((4,1,2),dtype=np.int32)
    add=points.sum(1)
    newPoints[0]=points[np.argmin(add)]
    newPoints[3]=points[np.argmax(add)]
    diff=np.diff(points,axis=1)
    newPoints[1]=points[np.argmin(diff)]
    newPoints[2]=points[np.argmax(diff)]
    return newPoints
def splitImg(image):
    rows=np.vsplit(image,9)
    boxes=[]
    for row in rows:
        cols=np.hsplit(row,9)
        for col in cols:
            boxes.append(col)
    return boxes
def displayNumbers(img,numbers,color=(50,205,50)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,color,1,cv2.LINE_AA)
    return img
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
    
####################################### Digit Prediction ###########################################################

def loadMyModel():
    myModel=load_model('CNNModel.h5')
    return myModel
def calProb(ypred):
    probablities=[]
    ypredict=np.argmax(ypred,axis=1)
    for i in ypred:
        probablities.append(i[np.argmax(i)])
    yprob=np.array(probablities)
    return yprob,ypredict

####################################### Sudoku Solver ##############################################################

def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]== 0):
                l[0]= row
                l[1]= col
                return True
    return False
def used_in_row(arr, row, num):
    for i in range(9):
        if(arr[row][i] == num):
            return True
    return False
def used_in_col(arr, col, num):
    for i in range(9):
        if(arr[i][col] == num):
            return True
    return False
def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if(arr[i + row][j + col] == num):
                return True
    return False
def check_location_is_safe(arr, row, col, num):
    return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 3,col - col % 3, num)
def solve_sudoku(arr):    
    l =[0, 0]   
    if(not find_empty_location(arr, l)):
        return True
    row = l[0]
    col = l[1]
    for num in range(1, 10):
        if(check_location_is_safe(arr,row, col, num)):
            arr[row][col]= num
            if(solve_sudoku(arr)):
                return True
            arr[row][col] = 0      
    return False
