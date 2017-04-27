#coding=utf-8
import os
from matplotlib import pyplot as plt
import shutil
caffe_root = '/home/hjimce/tools/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import  cv2
import  dlib
import numpy as np
import  time
import random
#根据人脸框bbox，从一张完整图片裁剪出人脸,并保存问文件名cropimgname
#如果未检测到人脸,那么返回false,否则返回true
#加载训练好的模型
def loadmodel():


	mean_filename='model/imagenet_mean.binaryproto'
	gender_net_pretrained='model/1.caffemodel'
	gender_net_model_file='model/deploy_gender.prototxt'




	proto_data = open(mean_filename, "rb").read()
	a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	mean  = caffe.io.blobproto_to_array(a)[0]


	gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,mean=mean,
					   channel_swap=(2,1,0),
					   raw_scale=255,
					   image_dims=(90, 90))
	return  gender_net

#用于机器预挑选
def batchclassify(filepath='1'):

	pickfile=filepath+'pick'
	os.mkdir(pickfile)
	model=loadmodel()
	imglists=os.listdir(filepath)
	for imgpath in imglists:

		input_image = caffe.io.load_image(filepath+'/'+imgpath)
		prediction_gender=model.predict([input_image],False)
		index=prediction_gender[0].argmax()
		newpath=pickfile+'/'+str(index)
		if os.path.exists(newpath) is False:
			os.makedirs(newpath)
		newname=newpath+'/'+imgpath
		shutil.copy(filepath+'/'+imgpath,newname)




#predict('1.jpg')
batchclassify('photo_03_7397')
#accurate()
#testonefile()







