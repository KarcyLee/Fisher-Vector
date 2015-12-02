# Fisher-Vector
Thanks to https://github.com/jacobgil/pyfishervector

本程序主要用来实现fisher vector 编码。
对我而言，想要一个干净的不局限于SIFT或图像的编码器，输入为需要编码的词向量，输出为编码后的fisher vector 向量。

Jacobgil提供了很好的示范。但是，封装与图像结合过密，训练GMM模型速度慢。
我采用了sklearn的GMM模型代替cv的EM算法。同时在归一化Fisher vector时做了改动：权值、均值、方差分布squre归一化。

Usage:
1: 读入数据
read_featuers:  读入数据成为一个numpy矩阵，每行为一个样本。
generate_gmm:
生成GMM模型。输入为Descriptors，为训练数据（词向量矩阵），每行为一个单词。folder 为存储GMM模型路径。N为聚类中心数目，在这里为混合高斯模型中高斯模型的个数。
load_gmm:
加载gmm模型
fisher_vector:
输入为Samples,若干个待编码的词向量构成的矩阵。输出为一个fv向量。


