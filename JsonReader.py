import json
import os

class JsonReader:
    def __init__(self):
        pass

    def readFile(self,file_path):
        """
        file_path为文件名+路径
        返回该json文件解析的dict
        """
        with open(file_path,'r') as load_f:
            load_dict = json.load(load_f)
        return load_dict
    
    def readDir(self,dir_path,encoding = 'utf-8'):
        """
        dir_path为文件夹路径
        encoding为解析方式
        返回一个dict[文件名(不包含后缀名)] = {解析数据}
        """
        json_file_names = self.__getJsonFileNames(dir_path)
        result = dict()
        for json_file_name in json_file_names:
            with open(file = dir_path + '//' + json_file_name + '.json',mode = 'r',encoding = encoding) as load_f:
                result[json_file_name] = json.load(load_f)
        return result

    def __getJsonFileNames(self,dir_path):
        """
        获取dir_path下的所有json文件名(不包含后缀)
        """
        json_file_names = list()
        for root, dirs, files in os.walk(dir_path):  
            for file_ in files:  
                if os.path.splitext(file_)[1] == '.json':  
                    json_file_names.append(os.path.splitext(file_)[0])  
        return json_file_names

if __name__ == '__main__':
    reader = JsonReader()
    print(reader.readDir('data//jsondata'))