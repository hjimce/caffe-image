---
title: caffe image项目训练流程 
---
-	运行packdata模块中的makelist.py：spiltdata函数调用，其对图片进行名字进行处理，防止图片名中有空格，并把训练数据且分成训练数据集、验证数据集两部分数据放在train、val文件夹
-	运行precrop模块，分别对train、val文件夹图像进行标准裁剪，比如裁剪出一张人脸图片中的嘴巴，可以运行crop_mouth_rect.py
-	运行packdata模块中的augment.py，对文件train夹中的数据进行augment。
-	运行packdata模块中的makelist.py：调用writetrainlist('train')、writetrainlist('val')用于生成标签列表文件.txt	
-	运行packdata模块中的creat_imgnet.sh用于把数据打包成lmdb格式文件
-	运行packdata模块中的make_imagenet_mean.sh用于计算lmdb数据的均值文件
-	编写网络结构、训练配置等文件，运行run.sh训练起来，训练完毕后，拿出模型、均值文件
-	运行test模块中的testone.py，用于预测查看结果
-	运行parse_log.py用于查看训练曲线图

