import os
import json

class BatchRecorder:
        def __init__(self, record_path, record_file_name = 'record.json'):
            self.record_file_name = record_file_name
            self.record_path = record_path
            if not os.path.exists(self.record_path):
                os.makedirs(self.record_path)

        def read(self):
            """
            读取record
            返回一个保存了结果的字典
            {tfrecords_name:{'num':num}}
            """
            if not os.path.exists(self.record_path + '\\' + self.record_file_name):
                return dict()
            with open(self.record_path + '\\' + self.record_file_name ,'r') as load_f:
                data = json.load(load_f)
            return data

        def reflash(self,json_data):
            """
            更新文件
            """
            with open(self.record_path + '\\' + self.record_file_name, 'w') as load_f:
                json.dump(json_data, load_f)