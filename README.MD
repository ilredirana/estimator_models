## Some models implement with Tensorflow high level API Estimator
### 一些使用 Tensorflow 高级 API Estimator 实现的模型


#### 1. MNIST


这是最简单的CNN模型。

`mnist_model.py`实现了 MNIST 的 Estimator 代码。
`transfer_and_finetune.py`实现了基于前者训练好的参数，再进行迁移学习的 Estimator，代码中将其体现为迁移到Hnd首写字母训练集上


#### 2.Inception_v3

这是进阶的模型。

根目录有两个模型，分别是 Keras 实现的 Inception_v3 和 Tensorflow layers API 实现的 Inception_v3 的 Estimator。

`keras_to_estimator`目录包括了将 Keras 的 Inception_v3 模型转换为 Estimator，迁移学习，保存和加载的方法。




#### 3.SSD目标检测

这是再一步进阶的模型。

##### TODO:
- 整理好代码，多写一些注释，上传



------

欢迎Star和Issue哇 ^3^
