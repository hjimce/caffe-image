#coding=utf-8
#拷贝文件到另外一个文件
import  os
import  shutil
racecrop=['pickphoto_09//black','pickphoto_09//white','pickphoto_09//yellow','pickphoto_09//brown']
raceall=['allpick//black','allpick//white','allpick//yellow','allpick//brown']
for crop,all in zip(racecrop,raceall):
    de=os.listdir(crop)
    for img in de:
        path='photo_09'+'//'+img
        newpath=all+'//'+img
        if os.path.exists(path):
            shutil.copy(path,newpath)
#求取两个文件夹差异文件
'''import  os
de=os.listdir('yellowpick//'+'delete')
for img in de:
    path='yellowpick//'+'ori//'+img
    if os.path.exists(path):
        os.remove(path)'''