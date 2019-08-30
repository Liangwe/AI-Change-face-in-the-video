import os
import cv2
from PIL import Image  
def size(transferpath):#获取图片像素的大小
    filelist = os.listdir(transferpath)
    
    img = Image.open(transferpath + filelist[0])  
    return  img.size
def mergevideo(transferpath):
    img_root = transferpath #这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
    fps = 24    #保存视频的FPS，可以适当调整
    filelist = os.listdir(transferpath)  #得到所有帧的文件名，在循环中使用到文件数目

    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    s=size(transferpath) #获取图片像素的大小
    videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,s)#最后一个是保存图片的尺寸

    for i in range(len(filelist)):
        frame = cv2.imread(img_root+str(i+1)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()
