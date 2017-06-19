#coding=utf-8
import os
from matplotlib import pyplot as plt
import shutil
caffe_root = '/home/research/tools/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import  cv2
import  dlib
import numpy as np
import  time
import random
from  multiprocessing  import  Pool
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
	miny=int(max(0,y-scale*h))
	minx=int(max(0,x-scale*w))
	maxy=int(min(height,y+(1+scale)*h))
	maxx=int(min(weight,x+(1+scale)*w))
	roi=img[miny:maxy,minx:maxx]
	rectshape=roi.shape
	maxlenght=max(rectshape[0],rectshape[1])
	img0=np.zeros((maxlenght,maxlenght,3))
	img0[int(maxlenght*.5-rectshape[0]*.5):int(maxlenght*.5+rectshape[0]*.5),int(maxlenght*.5-rectshape[1]*.5):
	int(maxlenght*.5+rectshape[1]*.5)]=roi

	cv2.imwrite(cropimgname,img0)

	return  True
def predict(model,imgpaths,bbox=None):

	result=[]
	gender_listc=['7','10','9','8','3','4','1','6','5','2']
	#gender_listc=['黑种人','棕种人','白种人','黄种人']
	for i,imgpath in enumerate(imgpaths):
		cropimgname='crop//'+os.path.basename(imgpath)
		if getface(imgpath,cropimgname)==False:
			result.append([None,None,cropimgname])
		else:
			t0 = time.clock()
			input_image = caffe.io.load_image(cropimgname)
			prediction_gender=model.predict([input_image],True)
			print (time.clock()-t0)*1000
			propra={}
			for i in range(len(gender_listc)):
				propra[gender_listc[i]]=int(prediction_gender[0][i]*100)
			propra= sorted(propra.iteritems(), key=lambda d:d[1],reverse=True)#字典根据值排序
			prostr=''
			for key in propra:
				prostr=prostr+'\t'+key[0]+':'+str(key[1])+'%'+'\t'
			result.append([propra[0][0],prostr,cropimgname])
	return  result
#加载训练好的模型
def loadmodel(pretrained_model='train.caffemodel',network='deploy.prototxt',size=256):

	gender_net = caffe.Classifier(network, pretrained_model,
					   channel_swap=(2,1,0),
					   raw_scale=255,
					   image_dims=(size, size))
	return  gender_net


model=loadmodel('model2.0/train.caffemodel','model2.0/net.prototxt',90)
#用于测试精度
def accurate(filepatht='val'):
	gender_listc=['7','10','9','8','3','4','1','6','5','2']



	error_count=0
	all_count=0
	for race in gender_listc:
		pickpatch="val_pick/"+race
		os.makedirs(pickpatch)
		filepath=filepatht+'/'+race
		imglists=os.listdir(filepath)
		random.shuffle(imglists)
		for imgpath in imglists:
			[[prace,racepro,cropimg]]=predict(model,[filepath+'/'+imgpath])
			if prace is not None:

				all_count+=1
				if prace!=race:
					shutil.copy(filepath+'/'+imgpath,pickpatch+'/pre:'+str(prace)+"_id"+str(all_count)+'.jpg')
					'''resized_image = cv2.resize(cv2.imread(cropimg), (500, 500))
					cv2.imshow('pre:'+prace+'\t'+'ac:'+race,resized_image)
					cv2.moveWindow('pre:'+prace+'\t'+'ac:'+race, 100, 100);
					cv2.waitKey(0)'''
					error_count+=1
			else:
				print 'detect no face'
	print 'accurate:',1-error_count/float(all_count)






#用于机器预挑选
def batchclassify(filepath='test'):
	pickfile=filepath+'pick'
	os.mkdir(pickfile)

	imglists=os.listdir(filepath)

	oldpath=[(filepath,pickfile,imagepath) for imagepath in imglists]
	#for l in oldpath:
	#	oneclassify(l)
	pool=Pool()
	pool.map(oneclassify,oldpath)








#predict('1.jpg')
#batchclassify('photo8_front')
accurate()
#testonefile()







