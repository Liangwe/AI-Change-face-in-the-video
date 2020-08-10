# AI-Change-face-in-the-video
将视频中的人脸更换为指定照片中的人脸，并且输出视频。

Dlib的安装  pip install dlib==19.6.1  这种最简单

预先训练好的模型，人脸库下载-> <a href="http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
">链接</a>

trump.mp4为被更换的视频，大家如果改换视频的话，代码中视频名称也需要更改。
1.jpg为替换照片，注意此张照片和视频中只能有一个面部，否则会报错。
由于人脸差异的原因，结果可能不会太理想，所以尽量找肤色相同，面容相似的两个人效果会好很多
