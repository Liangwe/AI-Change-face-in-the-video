from transfer import *
from newvideo import mergevideo
from PythonCv2 import video2img
from deletecache import del_file

import os 

count = 1

originpath = 'origin/'
transferpath = 'transfer/'  #原始帧的路径和转换后帧的路径 一定要加 / 
videofilepath = 'trump.mp4'    #文件路径可自己定义

video2img(videofilepath,originpath)

print('视频的每一帧分离完成，开始转换')
filelist=os.listdir(originpath)
num = len(filelist)

newlist = os.listdir(originpath)

print('视频一共有：'+ str(num) + '帧')

im2, landmarks2 = read_im_and_landmarks("1.jpg")  #人物模型，将要替换到视频中的人
                                            #在同级目录下，如果更换，请定义路径
                        #写到for循环外边 减少了建立模型的时间，稍微快了那么一丢丢

for i in newlist:  #获得文件数目进行逐帧转换
    
    im1, landmarks1 = read_im_and_landmarks(originpath + i )

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colors(im1, warped_im2, landmarks1,landmarks2)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    cv2.imwrite(transferpath + i, output_im)

    count = count + 1
    if (count%100 ==0):
        print('已完成'+str(count)+'帧')

print('逐帧转换完成，开始合并视频：')

mergevideo(transferpath)

print('视频转换完成，请确认是否清除图片缓存')

m = input('是否删除缓存 y/n')

if (m == 'y'):
    del_file(originpath)
    del_file(transferpath)
    print('删除成功')
else:
    print('缓存未删除')

print ('Done！！！')

