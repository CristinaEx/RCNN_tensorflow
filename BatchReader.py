from BatchRecorder import BatchRecorder
from PIL import Image
import tensorflow as tf
import os
import numpy

class BatchReader:
    def __init__(self):
        pass

    def readAndDecordBatch(self,path_file_name):
        """
        path_file_name:batch的文件名+路径
        返回:
        img:reshape(h,w,mod,things_num)后的图像数据
        label:int数据，标号
        """
        # 根据文件名生成一个队列   
        filename_queue = tf.train.string_input_producer([path_file_name])   
        reader = tf.TFRecordReader()  
        _, serialized_example = reader.read(filename_queue)     
        # 返回文件名和文件 
        features = tf.parse_single_example(serialized_example,features={      
                                               'label': tf.FixedLenFeature([], tf.int64),                                                                 
                                               'img_raw' : tf.FixedLenFeature([], tf.string),}) 
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        # 获取数据集信息
        rec_reader = BatchRecorder(os.path.dirname(path_file_name))
        mes = rec_reader.read()
        # reshape 
        img = tf.reshape(tensor = img,shape = mes[os.path.basename(path_file_name)]['shape'])
        img = tf.cast(img,tf.float32)
        # 在流中抛出label张量
        label = tf.cast(features['label'], tf.int32)
        return img, label

    def getTrainBatch(self,path_file_name,num = 1):
        """
        path_file_name:path + file_name
        return:
        img_batch:图像数据流
        label_batch:标签数据流
        """
        # 获取数据集信息
        rec_reader = BatchRecorder(os.path.dirname(path_file_name))
        mes = rec_reader.read()[os.path.basename(path_file_name)]
        print(mes)
        # tf.train.batch()按顺序读取数据
        img, label = self.readAndDecordBatch(path_file_name)
        img_batch, label_batch = tf.train.batch([img,label], batch_size = num, capacity = mes['num'])
        return img_batch, label_batch

def test(path):
    img_batch, label_batch = BatchReader().getTrainBatch(path_file_name = path)
    mes = BatchRecorder(os.path.dirname(path)).read()[os.path.basename(path)]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        img,label = sess.run([img_batch,label_batch])
        img = numpy.array(img,dtype = 'uint8')
        img = numpy.reshape(img,mes['shape'])
        print(label)
        Image.fromarray(img,mode = 'RGB').show()
        try:
            # 请求线程终止
            coord.request_stop()
        except tf.errors.OutOfRangeError:  
            print ('Done training -- epoch limit reached') 
        finally:
            # 请求线程终止
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    reader = BatchReader()
    reader.getTrainBatch(path_file_name = 'train\\train.tfrecords')
    # test('train\\train.tfrecords')