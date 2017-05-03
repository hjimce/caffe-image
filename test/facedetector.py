#coding=utf-8
import  cv2
import  dlib
import numpy as np
#根据人脸框bbox，从一张完整图片裁剪出人脸
def getface(imgpath):
    bgrImg = cv2.imread(imgpath)
    print bgrImg.shape
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    detector=dlib.get_frontal_face_detector()
    #img = io.imread('1.jpg')
    faces = detector(rgbImg, 1)
    if len(faces) <=0:
        return
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
    print img0
    cv2.imwrite(imgpath,img0)
#getface('1.jpg')