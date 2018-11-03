import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE=30
SEED=2

rdm=np.random.RandomState(seed=SEED)
X=rdm.randn(300,2)
#作为输入数据集的标签
Y_=[int(x0*x0+x1*x1<2) for (x0,x1)in X]
Y_c=[['red' if y else 'blue']for y in Y_]
#把X整理成n行2列，把Y_整理成n行-列
X=np.vstack(X).reshape(-1,2)
Y_=np.vstack(Y_).reshape(-1,1)#标准答案
print(X)
print(Y_)
print(Y_c)
#画出各行的(x0,x1)
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))#取X中第一列（横坐标）第二列（纵坐标）
plt.show()#画出散点

#定义神经网络的输入，参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)#随机设置参数w
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))#对w的平方求和，l2正则化,定义正则化
    return w
def get_bias(shape):
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=get_weight([2,11],0.01)#w1 2*11
b1=get_bias([11])#11个
y1=tf.nn.relu(tf.matmul(x,w1)+b1)#第一层,激活传值

w2=get_weight([11,1],0.01)
b2=get_bias([1])#偏置
y=tf.matmul(y1,w2)+b2//输出层

#定义损失函数
loss_mse=tf.reduce_mean(tf.square(y-y_))#交叉熵
loss_total=loss_mse+tf.add_n(tf.get_collection('losses'))#正则化损失函数
'''
#定义反向传播方法：不含正则化
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=40000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%300
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y:Y_[start:end]})
        if i%2000==0:
            loss_mse_v=sess.run(loss_mse,feed_dict={x:X,y:Y_})
            print("after %d steps,loss is %f",(i,loss_mse_v))
    xx,yy=np.mgrid[-3:3:.01,-3:3:.01]
    grid=np.c_[xx.ravel(),yy.ravel()]#所以坐标点
    probs=sess.run(y,feed_dict={x:grid})
    probs=probs.reshape(xx.shape)#整理成和xx一样shape
    print("w1:\n",sess.run(w1))
    print("b1:\n",sess.run(b1))
    print("w2:\n",sess.run(w2))

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
'''
#定义反向传播方法:包含正则化
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss_total)#正则化，学习率为0.0001
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=40000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%300
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y:Y_[start:end]})
        if i%2000==0:
            loss_mse_v=sess.run(loss_mse,feed_dict={x:X,y:Y_})
            print("after %d steps,loss is %f",(i,loss_mse_v))
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]#组成坐标
    probs = sess.run(y, feed_dict={x: grid})#上色
    probs = probs.reshape(xx.shape)
    print("w1:\n", sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()


