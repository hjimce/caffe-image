import  shutil
import os
def cut_val(val_txt="../mutil_light/data/val.txt",dataroot='../mutil_light/data'):
	with open(val_txt,'r') as f:
		for l in f.readlines():
			oldpath=l.split()[0]
			oldpath=oldpath.replace('crop0','lacklight')
			oldpath=oldpath.replace('crop1','toolight')
			oldpath=oldpath.replace('crop2','topbottom')
			oldpath=oldpath.replace('crop3','rightlight')
			oldpath=oldpath.replace('crop4','leftlight')
			oldpath=oldpath.replace('crop5','goodlight')
			print oldpath
			oldpath_abs=os.path.join(dataroot,oldpath)
			newpath_abs=os.path.join(dataroot+'/val',oldpath)
			shutil.move(oldpath_abs,newpath_abs)
def cut(train_frontroot='../headangle/data/train/front',front_root='../headangle/d/front'):
	image=os.listdir(front_root)
	for i in image:
		oldpath=os.path.join(train_frontroot,i)
		if os.path.exists(oldpath):
			os.remove(oldpath)


cut_val()






