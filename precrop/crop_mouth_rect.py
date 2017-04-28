#coding=utf-8
import numpy as np
import  matplotlib.pyplot as plt
import  cv2
import  dlib
import  os
import shutil
from multiprocessing  import  Pool
#根据人脸框bbox，从一张完整图片裁剪出人脸,并保存问文件名cropimgname
#如果未检测到人脸,那么返回false,否则返回true
face_detector=dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#注意输入rect格式[x1,x2,y1,y2]
def get_imagerect(bgrImg,rect):
	img=bgrImg
	height, weight = np.shape(img)[:2]
	[x1, x2, y1, y2]=rect
	x = int(x1)
	y = int(y1)
	w = int(x2 - x1)
	h = int(y2 - y1)
	scaleh = 0.1
	scalew=0.0
	miny = int(max(0, y - scaleh * h))
	minx = int(max(0, x - scalew * w))
	maxy = int(min(height, y + (1 + scaleh) * h))
	maxx = int(min(weight, x + (1 + scalew) * w))
	roi = img[miny:maxy, minx:maxx]
	rectshape = roi.shape
	maxlenght = max(rectshape[0], rectshape[1])
	img0 = np.zeros((maxlenght, maxlenght, 3))
	img0[int(maxlenght * .5 - rectshape[0] * .5):int(maxlenght * .5 + rectshape[0] * .5),
	int(maxlenght * .5 - rectshape[1] * .5):int(maxlenght * .5 + rectshape[1] * .5)] = roi

	return  img0
	#plt.imshow(img0)
	#plt.show()
#返回一张图片,经过人脸特征点检测后,得到的左右眼睛的boundingbox
def get_mouthboudingbox(rgbimage):

	facesrects = face_detector(rgbimage, 1)
	if len(facesrects) <=0:
		return False
	facerect=max(facesrects, key=lambda rect: rect.width() * rect.height())

	shape = landmark_predictor(rgbimage, facerect)
	mouthlandmarks=[]
	for i in range(48,68):
		pt=shape.part(i)
		mouthlandmarks.append([pt.x,pt.y])
		#plt.plot(pt.x,pt.y,'ro')
		#plt.text(pt.x,pt.y,str(i))


	mouth_max=np.max(np.asarray(mouthlandmarks),axis=0)
	mouth_min=np.min(np.asarray(mouthlandmarks),axis=0)
	dic={}
	dic['mouth_rect']=[mouth_min[0],mouth_max[0],mouth_min[1],mouth_max[1]]

	'''right_eyelandmarks=[]
	for i in range(48,68):
		pt=shape.part(i)
		right_eyelandmarks.append([pt.x,pt.y])
		#plt.plot(pt.x,pt.y,'ro')
		#plt.text(pt.x,pt.y,str(i))
	#plt.imshow(rgbimage)
	#plt.show()'''
	return  dic
def mutil_thread(oldpath_newpath):
	oldname=oldpath_newpath[0]
	newname=oldpath_newpath[1]


	bgrImg = cv2.imread(oldname)
	if bgrImg is None:
		return
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	mouth_rect=get_mouthboudingbox(rgbImg)
	if mouth_rect is False:
		return

	cv2.imwrite(newname,get_imagerect(bgrImg,mouth_rect['mouth_rect']))


#两个眼睛分开训练
def getmouth_batch(filepath):

	pickfile=filepath+'rect'
	if os.path.exists(pickfile):
		shutil.rmtree(pickfile)
	clasiffys=os.listdir(filepath)
	for clasiffy in clasiffys:
		newpath=pickfile+'/'+clasiffy
		if os.path.exists(newpath) is False:
			os.makedirs(newpath)





		imglists=os.listdir(filepath+'/'+clasiffy)
		oldnames=[]
		newnames=[]
		for imgpath in imglists:
			oldname=filepath+'/'+clasiffy+'/'+imgpath
			newname=pickfile+'/'+clasiffy+'/'+imgpath
			oldnames.append(oldname)
			newnames.append(newname)

		pool=Pool()
		pool.map(mutil_thread,zip(oldnames,newnames))







getmouth_batch("../mouth/oridata")
