#coding=utf-8
import numpy as np
import  matplotlib.pyplot as plt
import  cv2
import  dlib
import  os
from multiprocessing import  Pool
#根据人脸框bbox，从一张完整图片裁剪出人脸,并保存问文件名cropimgname
#如果未检测到人脸,那么返回false,否则返回true
face_detector=dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def geteye_rect(imgpath):
	bgrImg = cv2.imread(imgpath)
	if bgrImg is None:
		return None
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	eyes_rect=get_eyeboudingbox(rgbImg)
	if eyes_rect is False:
		return None



	cv2.imwrite("left.jpg",get_imagerect(bgrImg,eyes_rect['left_rect']))
	cv2.imwrite("right.jpg",get_imagerect(bgrImg,eyes_rect['right_rect']))
	return  ["left.jpg","right.jpg"]
def geteye_images(imgpath):
	bgrImg = cv2.imread(imgpath)
	if bgrImg is None:
		return None
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	eyes_rect=get_eyeboudingbox(rgbImg)
	if eyes_rect is False:
		return None


	left=get_imagerect(bgrImg,eyes_rect['left_rect'])
	right=get_imagerect(bgrImg,eyes_rect['right_rect'])
	return  [left,right]
def getface_images(imgpath):
	print imgpath
	bgrImg = cv2.imread(imgpath)
	if bgrImg is None:
		return None
	print bgrImg.shape
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

	faces = face_detector(rgbImg, 1)
	if len(faces) <=0:
		return None
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
	if maxlenght<20:
		return  None
	img0=np.zeros((maxlenght,maxlenght,3))
	img0[int(maxlenght*.5-rectshape[0]*.5):int(maxlenght*.5+rectshape[0]*.5),
	int(maxlenght*.5-rectshape[1]*.5):int(maxlenght*.5+rectshape[1]*.5)]=roi
	return  img0
#注意输入rect格式[x1,x2,y1,y2]
def get_imagerect(bgrImg,rect):
	img=bgrImg
	height, weight = np.shape(img)[:2]
	[x1, x2, y1, y2]=rect
	x = int(x1)
	y = int(y1)
	w = int(x2 - x1)
	h = int(y2 - y1)
	scaleh = 0.4
	scalew=0.0
	miny = max(0, y - scaleh * h)
	minx = max(0, x - scalew * w)
	maxy = min(height, y + (1 + scaleh) * h)
	maxx = min(weight, x + (1 + scalew) * w)
	roi = img[miny:maxy, minx:maxx]
	rectshape = roi.shape
	maxlenght = max(rectshape[0], rectshape[1])
	img0 = np.zeros((maxlenght, maxlenght, 3))
	img0[(maxlenght * .5 - rectshape[0] * .5):(maxlenght * .5 + rectshape[0] * .5),
	(maxlenght * .5 - rectshape[1] * .5):(maxlenght * .5 + rectshape[1] * .5)] = roi

	return  img0
	#plt.imshow(img0)
	#plt.show()
#返回一张图片,经过人脸特征点检测后,得到的左右眼睛的boundingbox
def get_eyeboudingbox(rgbimage):

	facesrects = face_detector(rgbimage, 1)
	if len(facesrects) <=0:
		return False
	facerect=max(facesrects, key=lambda rect: rect.width() * rect.height())

	shape = landmark_predictor(rgbimage, facerect)
	left_eyelandmarks=[]
	for i in range(36,42):
		pt=shape.part(i)
		left_eyelandmarks.append([pt.x,pt.y])
		#plt.plot(pt.x,pt.y,'ro')
		#plt.text(pt.x,pt.y,str(i))


	left_max=np.max(np.asarray(left_eyelandmarks),axis=0)
	left_min=np.min(np.asarray(left_eyelandmarks),axis=0)
	dic={}
	dic['left_rect']=[left_min[0],left_max[0],left_min[1],left_max[1]]

	right_eyelandmarks=[]
	for i in range(42,48):
		pt=shape.part(i)
		right_eyelandmarks.append([pt.x,pt.y])
		#plt.plot(pt.x,pt.y,'ro')
		#plt.text(pt.x,pt.y,str(i))
	right_max=np.max(np.asarray(right_eyelandmarks),axis=0)
	right_min=np.min(np.asarray(right_eyelandmarks),axis=0)
	dic['right_rect']=[right_min[0],right_max[0],right_min[1],right_max[1]]



	#plt.imshow(rgbimage)
	#plt.show()
	return  dic
#两个眼睛分开训练
def geteye_batch(filepath):

	pickfile=filepath+'rect'
	os.mkdir(pickfile)
	clasiffys=os.listdir(filepath)
	for clasiffy in clasiffys:
		imglists=os.listdir(filepath+'/'+clasiffy)
		for imgpath in imglists:
			print imgpath
			oldname=filepath+'/'+clasiffy+'/'+imgpath

			bgrImg = cv2.imread(oldname)
			if bgrImg is None:
				continue
			rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
			eyes_rect=get_eyeboudingbox(rgbImg)
			if eyes_rect is False:
				continue

			newpath=pickfile+'/'+clasiffy
			if os.path.exists(newpath) is False:
				os.makedirs(newpath)
			newname_left=pickfile+'/'+clasiffy+'/'+'left'+imgpath
			cv2.imwrite(newname_left,get_imagerect(bgrImg,eyes_rect['left_rect']))
			newname_right=pickfile+'/'+clasiffy+'/'+'right'+imgpath
			cv2.imwrite(newname_right,get_imagerect(bgrImg,eyes_rect['right_rect']))
def crop_face(old_newpath):
	oldname=old_newpath[0]
	newname=old_newpath[1]

	face=getface_images(oldname)
	if face is None:
		return
	cv2.imwrite(newname,face)


#一整张人脸图片训练
def getface_batch(filepath):

	pickfile=filepath+'rect'
	os.mkdir(pickfile)
	clasiffys=os.listdir(filepath)
	oldpaths=[]
	newpaths=[]
	for clasiffy in clasiffys:
		imglists=os.listdir(filepath+'/'+clasiffy)
		for imgpath in imglists:
			oldname=filepath+'/'+clasiffy+'/'+imgpath
			newpath=pickfile+'/'+clasiffy
			if os.path.exists(newpath) is False:
				os.makedirs(newpath)
			newname_left=pickfile+'/'+clasiffy+'/'+imgpath
			oldpaths.append(oldname)
			newpaths.append(newname_left)
	#for newpath in zip(oldpaths,newpaths):
	#	crop_face(newpath)
	pool=Pool()
	pool.map(crop_face,zip(oldpaths,newpaths))






#geteyes_inone_batch('merge5.3_6.3')
#geteye_batch('merge1~5.1 6.1')
getface_batch("../faceshape/data/val")


#geteye_batch("../mouth/oridata")
