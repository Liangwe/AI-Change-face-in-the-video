#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import cv2
import os
def video2img(videofilepath, originpath):
    vc = cv2.VideoCapture(videofilepath)
    c = 0
    if vc.isOpened():
        rval , frame=vc.read()
    else:
        rval = False

    timeF = 1  # 由于视频帧数过多 用此变量作为测试使用 正常使用时值为 1

    while rval:
        rval,frame = vc.read()
        if rval != False:  # 如果没有读到帧，跳出 break;
            if(c%timeF == 0):
                cv2.imwrite(originpath + str(c) + '.jpg' , frame)
            c=c+1
            cv2.waitKey(1)
        else:
            break

    vc.release()


if __name__ == "__main__":
    originpath = 'origin/'
    transferpath = 'transfer/'  # 原始帧的路径和转换后帧的路径 一定要加 /
    videofilepath = 'trump.mp4'  # 文件路径可自己定义

    video2img(videofilepath, originpath)