import os
from PIL import Image

class ImageReader:
    def __init__(self):
        pass
    
    def readImage(self,file_path):
        """
        读取一个图片
        返回Image对象
        """
        img = Image.open(file_path,mode = "r")
        return img

    def __getImageFileNames(self,dir_path):
        """
        获取文件夹路径下所有的图片文件的名字(包含后缀)
        """
        img_file_names = list()
        for root, dirs, files in os.walk(dir_path):  
            for file_ in files:  
                if os.path.splitext(file_)[1] == '.jpg' or os.path.splitext(file_)[1] == '.png':  
                    img_file_names.append(file_)
        return img_file_names

    def readImages(self,dir_path):
        """
        读取文件夹目录下的所有图片，返回
        dict[文件名(不包括后缀)] = img
        """
        img_file_names = self.__getImageFileNames(dir_path)
        result = dict()
        for img_file_name in img_file_names:
            img = Image.open(dir_path + '//' + img_file_name)
            result[img_file_name.split('.')[0]] = img
        return result