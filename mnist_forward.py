import tensorflow as tf

INPUT_NODE = 784#784 个像素点
OUTPUT_NODE = 10
LAYER1_NODE = 500#隐藏层500个节点

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))#随机生成参数w（满足截断式正态分布）
    #参数满足截断正态分布，并使用正则化，将每个参数的正则化损失加到总损失中
    if regularizer != None: #l2正则化
        #将每个参数参数的正则化损失加到总损失中tf.add_to_collection()表示将参数 w 正则化损失加到总损失 losses 中
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b
	
def forward(x, regularizer):
    #第一层：输入层784个节点，隐藏层500个节点
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])#由输入层到隐藏层的偏置 b1 形状为长度为 500的一维数组
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    #由隐藏层到输出层的偏置 b2 形状为长度为 10 的一维数组
    b2 = get_bias([OUTPUT_NODE])
    #由于输出 y 要经过 softmax 函数，使其符合概率分布，故输出 y 不经过 relu 函数
    y = tf.matmul(y1, w2) + b2
    return y
