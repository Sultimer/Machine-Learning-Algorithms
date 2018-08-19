import random
import numpy as np


#！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！！
#！！！！ 注释写的靠右   ！！！！
#！！！！ 请放大窗口阅读 ！！！！
#！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！！


class Network(object):

    #初始化
    def __init__(self, sizes):                                 
                                                                                    #sizes为列表类型，对应每层的神经元个数，第一层为输入层，不设置偏置
        self.num_layers = len(sizes)                                                #层数
        self.sizes = sizes                                                          #sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]                    #偏置随机初始，每层对应y行1列作为一个列表，输入层不计入内，存为列表类型
        self.weights = [np.random.randn(y, x)                   
                        for x, y in zip(sizes[:-1], sizes[1:])]                     #权重随机初始，总的列表中每个元素是一个列表，
                                                                                    #其中存放着对应两层之间的权重矩阵，也为列表类型，该列表内元素为元组类型

    #前向传播
    def feedforward(self, a):                                                       #前一级输出
        for b, w in zip(self.biases, self.weights):             
            a = sigmoid(np.dot(w, a)+b)                                             #计算当前输出
        return a                                                                    #前向传播


    #随机梯度下降算法
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        training_data = list(training_data)                                         #列表类型元组训练集
        n = len(training_data)

        if test_data:                                                               #测试集存在则初始化
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):                                                     #重复epochs次
            random.shuffle(training_data)                                           #打乱训练集
            mini_batches = [                     
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]                              #分为一定大小的小批量列表
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)                             #对每个小批量进行训练
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))                                #显示训练结果


    #更新权重
    def update_mini_batch(self, mini_batch, eta):                  
        nabla_b = [np.zeros(b.shape) for b in self.biases]                          #清零
        nabla_w = [np.zeros(w.shape) for w in self.weights]     
        for x, y in mini_batch:                                                     #对小批量列表中的所有小批量
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)                      #通过反向传播算法计算b,w的改变量
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]           #累加更新
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]           
        self.weights = [w-(eta/len(mini_batch))*nw                                  #由小批量计算公式更新到最终的矩阵中
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    #反向传播
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]                          #清零                       
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #前向传播
        activation = x
        activations = [x]                                                           #输入初始化
        zs = []                                                                     #用来存每个带权输入
        for b, w in zip(self.biases, self.weights):                                 
            z = np.dot(w, activation)+b                                             #计算新的带权输入
            zs.append(z)                                                            #添加新的带权输入到zs列表末尾
            activation = sigmoid(z)                                                 #计算输出
            activations.append(activation)                                          #添加新的输出到act列表末尾
        #后向传播
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])                                               #输出层误差计算
        nabla_b[-1] = delta                                                         #将输出层的偏置和权重写入
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):                                         
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp              #计算每层反向传播过去的delta
            nabla_b[-l] = delta                                                     #求偏导
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)                                                   #返回的是b和w的改变量的矩阵

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)                         #列表存放输入x实际输出的最高激活值所分类的y'和y的元组类型
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)                          #返回正确的数量

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)                                               #计算代价
 
#数学计算
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))                            #sigmoid函数返回值

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))                       #sigmoid求导返回值
