import os
import time
import tensorflow as tf
import random
from DataReader import DataReader
from BatchRecorder import BatchRecorder

class BatchMaker:
    def __init__(self):
        pass
    
    def __CheckPath(self,path):
        """
        若path不存在，则创建它
        """
        if os.path.exists(path) == False:
            os.mkdir(path)

    def make(self,img_path,json_path,k,output_path,output_name):
        """
        img_path为image的路径
        json_path为图像的json标注数据的路径
        (m,n) = k
        w,h = img.size
        data.shape = (int(w/m)*m,int(h/n)*n)
        output_path为输出路径
        """
        datas = DataReader().readData(img_path,json_path,k)
        writer = tf.python_io.TFRecordWriter(output_path + '//' + output_name)
        for data in datas:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['index']])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['image'].tobytes()]))
                    }
                )
            )
            writer.write(example.SerializeToString())
        recorder = BatchRecorder(output_path)
        record_data = recorder.read()
        record_data[output_name] = dict()
        record_data[output_name]['num'] = len(data.keys())
        m,n = k
        record_data[output_name]['shape'] = [m,n,3]
        recorder.reflash(record_data)

    def makeTwoRandomOutput(self,img_path,json_path,k,output_train,output_test,ratio):
        """
        img_path为image的路径
        json_path为图像的json标注数据的路径
        (m,n) = k
        w,h = img.size
        data.shape = (int(w/m)*m,int(h/n)*n)
        output为输出路径
        ratio为train数量:test数量
        """
        random.seed(int(time.time()))
        datas = DataReader().readData(img_path,json_path,k)
        writer_first = tf.python_io.TFRecordWriter(output_train)
        writer_second = tf.python_io.TFRecordWriter(output_test)
        ratio = 1 - 1 / (ratio + 1)
        num_first = 0
        num_second = 0
        m,n = k
        for data in datas:
            if random.random() <= ratio:
                writer = writer_first
                num_first += 1
            else:
                writer = writer_second
                num_second += 1
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['index']])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['image'].tobytes()]))
                    }
                )
            )
            writer.write(example.SerializeToString())
        if num_first != 0:
            recorder_first = BatchRecorder(os.path.dirname(output_train))
            output_train = os.path.basename(output_train)
            record_data_first = recorder_first.read()
            record_data_first[output_train] = dict()
            record_data_first[output_train]['num'] = num_first
            record_data_first[output_train]['shape'] = [m,n,3]
            recorder_first.reflash(record_data_first)
        if num_second != 0:
            recorder_second = BatchRecorder(os.path.dirname(output_test))
            output_test = os.path.basename(output_test)
            record_data_second = recorder_second.read()
            record_data_second[output_test] = dict()
            record_data_second[output_test]['num'] = num_second
            record_data_second[output_test]['shape'] = [m,n,3]
            recorder_second.reflash(record_data_second)

if __name__ == '__main__':
    BatchMaker().makeTwoRandomOutput('data','data//jsondata',(224,224),'train//train.tfrecords','test//test.tfrecords',4)