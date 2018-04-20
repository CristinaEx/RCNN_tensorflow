from BatchReader import BatchReader
from BatchRecorder import BatchRecorder
import tensorflow as tf
import os
import numpy
# 防跳出警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# VGG16网络模型
class VGG16model:
    def __init__(self):
        pass

    def getWeightsAndBiases(self, shape, predict_num = 10, div = 100):
        """
        shape = [w,h,mod]
        w,h需为32倍数
        predict_num:预测的类别数量
        div:权值和偏值 /= div(防止梯度爆炸)
        获取随机的VGG16的权重和偏值
        """
        weights = {  
            # 3x3 conv, 3 input, 24 outputs  
            'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),  
            'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64])),  
            # pool
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),  
            'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128])),  
            # pool  
            'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256])),  
            'wc6': tf.Variable(tf.random_normal([3, 3, 256, 256])),  
            'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256])),  
            # pool
            'wc8': tf.Variable(tf.random_normal([3, 3, 256, 512])),  
            'wc9': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc10': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            # pool
            'wc11': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc12': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512])),  
            # fully connected, 32*32*96 inputs, 1024 outputs  
            'wd1': tf.Variable(tf.random_normal([int(shape[0] * shape[1] / 2), 1024])),  
            'wd2': tf.Variable(tf.random_normal([1024, 1024])),  
            # 1024 inputs, predict_num outputs (class prediction)  
            'out': tf.Variable(tf.random_normal([1024, predict_num]))
        }
        biases = {  
            'bc1': tf.Variable(tf.random_normal([64])),  
            'bc2': tf.Variable(tf.random_normal([64])),  
            # pool
            'bc3': tf.Variable(tf.random_normal([128])),  
            'bc4': tf.Variable(tf.random_normal([128])),
            # pool
            'bc5': tf.Variable(tf.random_normal([256])),  
            'bc6': tf.Variable(tf.random_normal([256])),  
            'bc7': tf.Variable(tf.random_normal([256])), 
            # pool
            'bc8': tf.Variable(tf.random_normal([512])),  
            'bc9': tf.Variable(tf.random_normal([512])),  
            'bc10': tf.Variable(tf.random_normal([512])), 
            # pool
            'bc11': tf.Variable(tf.random_normal([512])),  
            'bc12': tf.Variable(tf.random_normal([512])),  
            'bc13': tf.Variable(tf.random_normal([512])), 
            # fully connected, 32*32*96 inputs, 1024 outputs  
            'bd1': tf.Variable(tf.random_normal([1024])),  
            'bd2': tf.Variable(tf.random_normal([1024])), 
            # 1024 inputs, predict_num outputs (class prediction)  
            'out': tf.Variable(tf.random_normal([predict_num]))  
        }
        # 防止梯度爆炸
        for key in weights.keys():
            weights[key] = weights[key] / div
        for key in biases.keys():
            biases[key] = biases[key] / div
        return weights,biases

    def conv2d(self, data, weights, biase, strides=1): 
        """
        卷积函数设置:
        data:数据
        weights:权重
        biase:偏差值
        strides:步长
        返回:
        卷积->RELU->返回值
        """ 
        # Conv2D wrapper, with bias and relu activation  
        data = tf.nn.conv2d(data, weights, strides=[1, strides, strides, 1], padding='SAME')  
        data = tf.nn.bias_add(data, biase)  
        return tf.nn.relu(data)  

    def maxpool2d(self, data, k=2):  
        """
        池化:
        data:数据
        k:池化层大小(1,k,k,1)
        步长:(1,k,k,1)
        """
        # MaxPool2D wrapper  
        return tf.nn.max_pool(data, ksize=[1, k, k, 1], strides=[1, k, k, 1],  
                          padding='SAME')  

    def getNet(self, data, weights, biases, dropout):  
        """
        data:数据(128,128,3)
        weights:权重
        biases:偏置
        dropout:防止过度拟合(float32)
        """
        # Convolution Layer  
        conv1 = self.conv2d(data, weights['wc1'], biases['bc1'])  
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])  
        # Max Pooling (down-sampling)  
        pool1 = self.maxpool2d(conv2, k=2)  
        # print(pool1.shape) #(64,64,64)  
  
        # Convolution Layer  
        conv3 = self.conv2d(pool1, weights['wc3'], biases['bc3'])  
        conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'])  
        # Max Pooling (down-sampling)  
        pool2 = self.maxpool2d(conv4, k=2)  
        # print(pool2.shape) #(32,32,128)  
  
        # Convolution Layer  
        conv5 = self.conv2d(pool2, weights['wc5'], biases['bc5'])  
        conv6 = self.conv2d(conv5, weights['wc6'], biases['bc6'])  
        conv7 = self.conv2d(conv6, weights['wc7'], biases['bc7'])  
        # Max Pooling  
        pool3 = self.maxpool2d(conv7, k=2)  
        # print(pool3.shape) #(16,16,256)  
  
        # Convolution Layer  
        conv8 = self.conv2d(pool3, weights['wc8'], biases['bc8'])  
        conv9 = self.conv2d(conv8, weights['wc9'], biases['bc9'])  
        conv10 = self.conv2d(conv9, weights['wc10'], biases['bc10'])  
        # Max Pooling  
        pool4 = self.maxpool2d(conv10, k=2)  
        # print(pool4.shape) #(8,8,512)  
  
        conv11 = self.conv2d(pool4, weights['wc11'], biases['bc11'])  
        conv12 = self.conv2d(conv11, weights['wc12'], biases['bc12'])  
        conv13 = self.conv2d(conv12, weights['wc13'], biases['bc13'])  
        # Max Pooling  
        pool5 = self.maxpool2d(conv13, k=2)  
        # print(pool5.shape) #(4,4,512)  
  
        # Fully connected layer  
        # Reshape conv2 output to fit fully connected layer input  
        fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])  
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])  
        fc1 = tf.nn.relu(fc1)  
        # Apply Dropout  
        fc1 = tf.nn.dropout(fc1, dropout)  
  
        #fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])  
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])  
        fc2 = tf.nn.relu(fc2)  
        # Apply Dropout  
        fc2 = tf.nn.dropout(fc2, dropout)  
        ''''' 
        fc3 = tf.reshape(fc2, [-1, weights['out'].get_shape().as_list()[0]]) 
        fc3 = tf.add(tf.matmul(fc2, weights['out']), biases['bd2']) 
        fc3 = tf.nn.relu(fc2) 
        '''  
        # Output, class prediction  
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])  
        return out 

def runVGG16(path,predict_num):
    """
    使用vgg16模型训练数据
    predict_num:预测模型数量
    """
    # 获取样本集信息
    mes = BatchRecorder(os.path.dirname(path)).read()[os.path.basename(path)]
    img_batch, label_batch = BatchReader().getTrainBatch(path_file_name = path,num = mes['num'])
    # 把label_batch转换为onehot
    label_batch = tf.expand_dims(label_batch, 1)
    indices = tf.expand_dims(tf.range(0, mes['num'], 1), 1)
    concated = tf.concat([indices, label_batch],1)
    label_batch = tf.sparse_to_dense(concated, tf.stack([mes['num'], predict_num]), 1.0, 0.0)
    # 导入模型
    model = VGG16model()
    weights,biases = model.getWeightsAndBiases(shape = mes['shape'], predict_num = predict_num, div = 100)
    out = model.getNet(img_batch,weights,biases,0.2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=label_batch))  
    # Evaluate model  
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(label_batch, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    # 建立优化器 
    # 学习率逐渐下降
    global_step = tf.Variable(0.0, trainable = False)
    learning_rate = tf.train.exponential_decay(1000.0,global_step,10,0.95,staircase = True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss,global_step = global_step)
    # Initializing the variables  
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        most_ac = 0
        for i in range(3):
            print(str(i)+':')
            sess.run(train)
            ac = sess.run([accuracy])
            most_ac = max([most_ac,ac[0]])
            print('accuracy:'+str(ac))
            test = sess.run([loss])
            print('loss:'+str(test))
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
    runVGG16('train//train.tfrecords',3)