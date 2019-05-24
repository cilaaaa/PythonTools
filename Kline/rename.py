__author__ = 'Cila'
import os
def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir + "/"):
        if root == file_dir + "/":
            for file in files:
                L.append(os.path.join(root, file))
    return L

def loadFile(filePath):
    txts = file_name(filePath)
    for txt in txts:
        date = txt.split('-')[-1].split('.')[0]
        dir = txt.split('/')[0]
        os.renames(txt,dir + '/cfIF1811%' + date + '.csv')

loadFile('E:\\futureData\cfIF1811')