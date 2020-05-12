<<<<<<< HEAD
<<<<<< HEAD
###Pytorch-yolov3
###增加了视频处理
###增加了opencv处理
###因为是在别人的yolov3的开源上增加了cv2的部分，所以可能效率偏低
###因为在转化的时候需要将输入数据转化为
=======
=======
>>>>>>> cbe9eb438d608aa5038945672f038c228f19f8c9
2020.4.29  
基于 eriklindernoren/PyTorch-YOLOv3 修改的  
vedio基于opencv绘制 因为在修改原来的开源项目时 PIL数组和cv2数组通道顺序不一样，训练时没有修改，最后在显示图像的时候就要像转换格式，效率变低  
根据Issues修改了Model的变量类型警告  
根据opencv绘制了新的detect.py文件  
注释掉train.py的模型最终的评测部分 可能会引发bug，还没想好怎么改  
