import numpy
from ImageReader import ImageReader
from JsonReader import JsonReader

# 重新写，使用RCNN的思路
class DataReader:
    def __init__(self):
        pass

    def readData(self,img_path,json_path,k):
        """
        img_path为image的路径
        json_path为图像的json标注数据的路径
        data.shape = k
        返回
        list[
            {'index' = index,'image' = image},
            ...
        ]
        """
        result = list()
        imgs = ImageReader().readImages(dir_path = img_path)
        jsons = JsonReader().readDir(dir_path = json_path)
        for key in jsons.keys():
            if key in imgs.keys():
                img = imgs[key]
                for thing in jsons[key]["things"]:
                    img_now = img.crop((thing['x1'] ,thing['y1'] ,thing['x2'] ,thing['y2']))
                    result_now = dict()
                    result_now['index'] = thing['index']
                    result_now['image'] = img_now.resize(k)
                    result.append(result_now)
        return result

if __name__ == '__main__':
    DataReader().readData('data','data//jsondata',(224,224))