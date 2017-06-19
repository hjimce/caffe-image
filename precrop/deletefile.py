#coding=utf-8
import  shutil
import  os
def get_file(root):
    chairroot=os.listdir(root)
    chairroot=[os.path.join(root,c) for c in chairroot]
    files=[]
    for c in chairroot:
        files=files+os.listdir(c)
    return  files
def delete(root,files):
    print "origin files:",len(os.listdir(root))
    paths=[os.path.join(root,f) for f in files]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

    print "delete result:",len(os.listdir(root))



files=get_file('../mutil_light/data/train')#从这个目录中找出所有文件名
chairroot=os.listdir('../mutil_light/data/val')
for f in chairroot:
    valroot=os.path.join('../mutil_light/data/val',f)
    delete(valroot,files)#删除val中的所有与train名字相同的文件



