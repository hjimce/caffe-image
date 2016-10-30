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
face_detector=dlib.get_frontal_face_detector()
#caffe.set_mode_gpu()
def getface(imgpath,cropimgname):
	bgrImg = cv2.imread(imgpath)
	if bgrImg is None:
		return False
	print bgrImg.shape
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)


	#img = io.imread('1.jpg')
	faces = face_detector(rgbImg, 1)
	if len(faces) <=0:
		return False
	face=max(faces, key=lambda rect: rect.width() * rect.height())
	[x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
	img = bgrImg
	height, weight =np.shape(img)[:2]
	x=int(x1)
	y=int(y1)
	w=int(x2-x1)
	h=int(y2-y1)
	scale=0.4
	miny=max(0,y-scale*h)
	minx=max(0,x-scale*w)
	maxy=min(height,y+(1+scale)*h)
	maxx=min(weight,x+(1+scale)*w)
	roi=img[miny:maxy,minx:maxx]
	rectshape=roi.shape
	maxlenght=max(rectshape[0],rectshape[1])
	img0=np.zeros((maxlenght,maxlenght,3))
	img0[(maxlenght*.5-rectshape[0]*.5):(maxlenght*.5+rectshape[0]*.5),(maxlenght*.5-rectshape[1]*.5):(maxlenght*.5+rectshape[1]*.5)]=roi

	cv2.imwrite(cropimgname,img0)

	return  True
#加载训练好的模型
def loadmodel(mean_filename="mean.binaryproto",pretrained_model='train.caffemodel',network='deploy.prototxt',size=256):

	proto_data = open(mean_filename, "rb").read()
	a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	mean  = caffe.io.blobproto_to_array(a)[0]


	gender_net = caffe.Classifier(network, pretrained_model,mean=mean,
					   channel_swap=(2,1,0),
					   raw_scale=255,
					   image_dims=(size, size))
	return  gender_net

#用于测试精度
def accurate(filepatht='stdtest'):
	gender_listc=['black','brown','white','yellow']
	#gender_listc=['tt']
	gender_listc=['yellow']
	model=loadmodel()

	error_count=0
	all_count=0
	for race in gender_listc:
		filepath=filepatht+'/'+race
		imglists=os.listdir(filepath)
		random.shuffle(imglists)
		for imgpath in imglists:
			[[prace,racepro,cropimg]]=predict(model,[filepath+'/'+imgpath])
			if prace is not None:
				all_count+=1
				if prace!=race:
					resized_image = cv2.resize(cv2.imread(cropimg), (500, 500))
					cv2.imshow('pre:'+prace+'\t'+'ac:'+race,resized_image)
					cv2.moveWindow('pre:'+prace+'\t'+'ac:'+race, 100, 100);
					cv2.waitKey(0)
					error_count+=1
			else:
				print 'detect no face'
	print 'accurate:',1-error_count/float(all_count)
#用于机器预挑选
def batchclassify(filepath='test'):
	pickfile=filepath+'pick'
	os.mkdir(pickfile)
	model=loadmodel()
	imglists=os.listdir(filepath)
	count=0
	for imgpath in imglists:
		[[race,racepro,cropimg]]=predict(model,[filepath+'/'+imgpath])
		if race is not None:
			newpath=pickfile+'/'+race
			if os.path.exists(newpath) is False:
				os.mkdir(newpath)
			newname=newpath+'/'+imgpath
			shutil.copy(cropimg,newname)
		else:
			print 'detect no face'
def testonefile(dataroot='testimage'):
	listc=['black','brown','white','yellow']
	listcnew=['yellow','black','white','brown']
	pickfile=dataroot+'pick'
	for l in listc:
		os.makedirs(os.path.join(pickfile,l))
	imglists=os.listdir(dataroot)
	model_old=loadmodel('race/oldversion/mean.binaryproto','race/oldversion/train.caffemodel','race/oldversion/deploy.prototxt',256)
	model_new=loadmodel('race/newaccuracy93.5/t.binaryproto','race/newaccuracy93.5/t.caffemodel','race/newaccuracy93.5/deploy.prototxt',90)
	for imgpath in imglists:
		getface(os.path.join(dataroot,imgpath),'1.jpg')
		input_image = caffe.io.load_image('1.jpg')
		prediction_old=model_old.predict([input_image],False)
		maxpro_old=listc[prediction_old[0].argmax()]

		prediction_new=model_new.predict([input_image],False)
		maxpro_new=listcnew[prediction_new[0].argmax()]
		if maxpro_old!=maxpro_new:
			resized_image = cv2.resize(cv2.imread('1.jpg'), (500, 500))
			cv2.imshow('old:'+maxpro_old+'\t'+'new:'+maxpro_new,resized_image)
			cv2.moveWindow('pre:', 100, 100)
			cv2.waitKey(0)





		'''newpath=pickfile+'/'+maxpro+'/'+imgpath
		shutil.copy(os.path.join(dataroot,imgpath),newpath)'''




























#predict('1.jpg')
#batchclassify()
#accurate()
testonefile()







