import cv2
import os
def save_img(videofilepath,originpath):
    vc = cv2.VideoCapture(videofilepath)
    c=1
    if vc.isOpened():
        rval , frame=vc.read()
    else:
        rval = False

    timeF = 1  # 由于视频帧数过多 用此变量作为测试使用 正常使用时值为 1

    while rval:
        rval,frame=vc.read()
        if(c%timeF == 0):
            cv2.imwrite(originpath +str(c) + '.jpg' , frame)
        c=c+1
        cv2.waitKey(1)
    vc.release()


