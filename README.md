# Babysound-Sort-tensorflow
# 一、项目概述

本文是婴儿哭声分类识别系统化的主体部分，主要解决智能音频分类的问题。基于此目标，本文查找了大量资料，并做了大量实验，最后获得了一个婴儿哭声分类识别准确率相对较高的深度学习模型——迁移学习Urbansound数据模型。

本文将从问题出发，提出问题、分析问题、解决问题，讲解如何一步步地解决该问题，并最终获得满意的结果。

# 二、项目规划

## 1. 项目要点问题

* 婴儿哭声数据集——去哪找？
* 音频数据如何预处理？
* 音频如何分类？
* 神经网络结构如何设计？（DNN、DNN+CNN、迁移学习）
* 实验如何设计？
* .....

## 2. 项目开发工具

* 模型训练（GPU）——[Colab](https://colab.research.google.com/ "Colab")
* 本地环境管理——Anaconda
* 本地开发——Pycharm

## 3.项目涉及代码

* [DNN模型](https://github.com/ldknight/Babysound-Sort-tensorflow/blob/main/baby_sound_DNN.ipynb "DNN模型")；
* [DNN迁移Urbansound模型](https://github.com/ldknight/Babysound-Sort-tensorflow/blob/main/baby_sound_DNN_improve_urbansound.ipynb "DNN迁移Urbansound模型")；
* [CNN模型](https://github.com/ldknight/Babysound-Sort-tensorflow/blob/main/babysound_classification_tf_cnn.ipynb "CNN模型")；
* [CNN迁移Urbansound模型](https://github.com/ldknight/Babysound-Sort-tensorflow/blob/main/babysound_classification_tf_cnn_improve_urbansound.ipynb "CNN迁移Urbansound模型")；
* [迁移vggish模型](https://github.com/ldknight/Babysound-Sort-tensorflow/blob/main/babysound_classification_tf_cnn_improve_vggish.ipynb "迁移vggish模型")。

# 三、项目要点

## 1.婴儿哭声数据集

本项目遇到的第一个关键问题是数据集去哪里找？起初在网上找到一个[音频共享网站](https://freesound.org/search/?q=baby "音频共享网站")，在这里找到了很多婴儿哭声的音频，但是，数量依旧还是很有限的，而且音频分类的标签是否准确也具有一定的不确定性，所以转而去找有没有现成的婴儿哭声数据集。

各大机器学习比赛网站、各个数据集网站、国内外音频分类相关论文等。终于，在****[飞桨数据集](https://aistudio.baidu.com/aistudio/datasetdetail/41960/1 "飞桨数据集")****中找到了有标签的婴儿啼哭数据集。打开数据集可以看到train和test两个文件夹，如图1所示。由于test文件夹中的音频没有标签，所以只能使用train文件夹数据进行训练，数据集类别如图2所示。

![](https://img-blog.csdnimg.cn/a35e561883294adcaf10a9210a1a45b3.png)​

<p align="center">图1 数据集文件</p>

![](https://img-blog.csdnimg.cn/090fb82715d94b30a0186df3306b5a2f.png)​

<p align="center">图2 （train文件夹）训练集</p>
* 训练集中共有数据918条，音频格式为wav，数据长度5s~30s不等。

## 2.音频处理

### 2.1 音频问题

解决了数据集问题，下一步就需要考虑音频数据问题：

* **数据量**是否足够训练出符合要求的模型？
* **数据长度**是否符合神经网络需要？
* 数据应该 **如何送入网络** ？
* ......

### 2.2 解决音频问题

小编在研究生阶段研究的是图像处理，图像可以被看成是一个多维矩阵，具有长宽高，自然可以送入神经网络中，音频该如何处理呢？

![](https://img-blog.csdnimg.cn/img_convert/78f2ba88de712f2d2456be19bbb859ac.jpeg)​

查找资料发现，音频实际是以波的形式存在，若以时间为横轴、振幅为纵轴绘制图像，则如图3所示。

![](https://img-blog.csdnimg.cn/e2bbd15c0ca440aca680679c2fdffc5a.png)​

<p align="center">图3 波形图</p>

通过查看原始音频文件数据存储发现，虽然音频文件同样可表现为张量数据（如图4所示），但在深度学习方面通常不会将其直接放入网络，通常的做法是将音频转换为频谱图（如图5所示）。频谱图是音频波的简洁“快照”，因为它是图像，所以非常适合输入到为处理图像而开发的基于CNN的架构中。图6展示了音频深度学习流程图。

![](https://img-blog.csdnimg.cn/img_convert/a10ff0e75e3db561c24321e52f81a42a.png)​

<p align="center">图4 音频文件</p>

![](https://img-blog.csdnimg.cn/5a82fd63e81447cf913300c42c9fc6b7.png)​

<p align="center">图5 频谱图（spectrogram）</p>

![](https://img-blog.csdnimg.cn/img_convert/ba40188817ef0f6c2070513f53b16cfe.png)​

<p align="center">图6 音频深度学习流程图</p>

## 3.深度学习网络结构

针对本项目，小编考虑了几种可能的深度学习网络结构，如全连接神经网络（DNN）、卷积神经网络（CNN+DNN）、迁移学习（[vggish](https://github.com/luuil/Tensorflow-Audio-Classification "vggish")）。

## 4.音频数据

* 婴儿哭声音频

大家不难发现，其实，本文所使用的婴儿哭声数据集并不是标准的数据集，而且数据量也很小，在判断网络模型优劣方面并不具有权威性，所以本文将使用婴儿哭声测试集作为评估模型的一个参考，同时使用标准音频分类数据集（[Urbansound8K](https://urbansounddataset.weebly.com/urbansound8k.html "Urbansound8K")）对模型进行辅助评估，并最终找到识别效果较好的深度学习分类模型。

* 实验数据集

`本项目希望音频的最大长度不超过10s，故针对所使用的数据需要经过裁剪处理，并映射到指定长度。`

Urbansound8K是包含10类声音的数据集，包括狗叫、汽车喇叭等声音，音频长度均在10s以内；

Babysound是一类包含6类声音的数据集，包括饿了、想睡觉等声音，音频长度为5s～30s不等（如图7所示），本项目将通过裁剪、分割不符合条件的音频，最终音频如图8所示。

![](https://img-blog.csdnimg.cn/2b17795c1bbf496d8d703ddff3c582de.png)​


<p align="center">图7 原始婴儿哭声音频</p>

![](https://img-blog.csdnimg.cn/9e42e5d05881413baae15e614c33cd11.png)​



<p align="center">图8 分割之后的婴儿哭声音频数据</p>

* 关于婴儿哭声数据集的测试集

由于分割后的婴儿哭声数据集数量依然很小（共2184条），并不足以满足训练集、验证集、测试集的各部分需求，所以本项目采用随机从**原始婴儿哭声数据集**中（长度5s～30s的数据）抽取一定量的数据，并随机对抽到的每条数据进行裁剪，使得每条数据长度为10s，由此获得的有标签的294条音频数据将作为最终的婴儿哭声测试集。

# 四、实验

## 1. DNN网络

* DNN网络结构如图9所示；

![](https://img-blog.csdnimg.cn/99241ae66fbc4105948f157ea31ed607.png)​


<p align="center">图9 DNN网络结构</p>

* Urbansound数据集在该网络中的表现如图10所示；

![](https://img-blog.csdnimg.cn/df9b51ac4c5748b0815e1ef9d2c2f00e.png)​


<p align="center">图10 Urbansound数据集在DNN网络中的表现</p>

* 图11为Babysound数据集在该网络中的表现；

![](https://img-blog.csdnimg.cn/a41ee34798b148de87dddfef0a8d0e7f.png)​


<p align="center">图11 Babysound数据集在DNN网络中的表现</p>

* 图12为该模型在Babysound测试集上的准确率（0.8163265306122449）。

![](https://img-blog.csdnimg.cn/dabad947539f4b8e9126464747dd6073.png)​


<p align="center">图12 婴儿哭声测试集在该模型上的准确率</p>


## 2. DNN+CNN

* CNN卷积神经网络结构如图13所示；

![](https://img-blog.csdnimg.cn/fb7c2562df7a45339c36e4eb64827358.png)​


<p align="center">图13 CNN卷积神经网络结构</p>

* Urbansound数据在CNN网络中的表现如图14所示；

![](https://img-blog.csdnimg.cn/a0987d0f8c344c2cac870e1f78ef909e.png)​

<p align="center">图14 Urbansound数据在CNN网络中的表现</p>

* Babysound数据在CNN网络中的表现如图15所示；

![](https://img-blog.csdnimg.cn/1d285e4f40314fd692c0dcea82535681.png)​


<p align="center">图15 Babysound数据在CNN网络中的表现</p>

* CNN网络模型在Babysound测试上的准确率（0.9081632653061225）如图16所示；

![](https://img-blog.csdnimg.cn/f99582e9a9964ddb95484b42e0bade74.png)​


<p align="center">图16 Babysound测试集在CNN网络模型中的准确率</p>

* 在此网络结构下，通过迁移学习Urbansound数据集训练的模型，Babysound数据在此网络结构下的表现如图17所示；

![](https://img-blog.csdnimg.cn/61e242d3dd6c4638a54955e8d1f6ec0c.png)​


<p align="center">图17 Babysound数据在迁移学习Urbansound数据模型的表现</p>


* 迁移学习Urbansound模型在Babysound测试集上的准确率（0.9931972789115646）如图18所示。

![](https://img-blog.csdnimg.cn/f0e9e86250064546bf872514050ef91e.png)​


<p align="center">图18 Babysound测试集在迁移学习Urbansound模型上的准确率</p>


## 3. 迁移学习Vggish模型

[Vggish模型](https://github.com/tensorflow/models/tree/master/research/audioset/vggish "Vggish模型")是在YouTube的AudioSet数据预训练得到模型，VGGish支持从音频波形中提取具有语义的128维embedding特征向量，网络结构如图19所示。

![](https://img-blog.csdnimg.cn/8029132817784a308469100fb98e1ae4.png)​


<p align="center">图19 vggish网络结构</p>

在迁移学习vggish模型的基础上，实验在网络后添加长短时记忆网络(Long Short Term Memory Network, [LSTM](https://so.csdn.net/so/search?q=LSTM&spm=1001.2101.3001.7020 "LSTM"))和一个全连接，最终将婴儿哭声数据分成6类，网络结构如图20所示。

![](https://img-blog.csdnimg.cn/84af8d4152e2441399ffcc5e94c3034b.png)​



<p align="center">图20 在vggish网络后添加短时记忆网络并添加一层全连接</p>

图21展示了实验相关参数和实验结果；

![](https://img-blog.csdnimg.cn/c66d722e1c9a49c3ae45f5d691d248c8.png)​


<p align="center">图21 Babysound数据在迁移学习vggish网络模型上的表现</p>


图22为vggish迁移模型在Babysound测试集上的准确率（0.8639455782312925）。

![](https://img-blog.csdnimg.cn/010a074d8acf4bc3acfdd67d13358223.png)​


<p align="center">图22 Babysound测试集在迁移学习vggish模型上的准确率</p>


## 4.实验结果汇总

<p align="center">表1 各模型在Babysound测试集中的准确率</p>

| 深度网络                                                                | 在Babysound测试集中的准确率 |
| -----------------------------------------                             | -------------------------------------------- |
| DNN网络模型                                                        | 0.8163265306122449 |
| 迁移学习Urbansound的DNN网络模型              | 0.5578231292517006 |
| CNN网络模型                                                        | 0.9081632653061225 |
| 迁移学习Urbansound数据集的CNN网络模型   | **0.9931972789115646** |
| 迁移学习vggish模型                                              | 0.8639455782312925 |

---

# 五、总结与下期预告

本文所述实验内容是婴儿哭声分类识别系统的主体部分，该部分涉及5种网络模型，分别是 **DNN深度学习模型** 、 **迁移学习Urbansound的DNN网络模型** 、 **CNN深度学习模型** 、**迁移学习Urbansound数据模型**和 **迁移学习Vggish模型** ，其中迁移学习Urbansound数据模型在婴儿哭声测试集上的准确率最高，为 **0.9931972789115646** 。

在之后的学习中，小编会将完整婴儿哭声分类识别系统（如图23所示）发布，敬请期待：

* 微信小程序——微信搜索🔍“ **婴儿啼哭** ”或扫描图24；
* PC管理后台；
* 后端等。

![](https://img-blog.csdnimg.cn/6cc0f1a3f6734aa08a34132b3e85afe0.png)​


<p align="center">图23 完整的项目结构图</p>

![](https://img-blog.csdnimg.cn/32fba5ddccd349e9a077b90159a5aa6d.png)​


<p align="center">图24 婴儿啼哭小程序</p>

# 六、后记

项目开始于2022.03，从开始构思到完成本文提到的内容用时大概两周。本篇文章是我第一次尝试把做过的项目总结发布，一方面是为了让自己戒骄戒躁，另一方面是为了重拾信心。

回想做项目那几周真是难熬，项目截至日期的压力、身体还出现了问题，最难过的是这个时候最值得信赖的情感也出现了问题，失眠、抑郁、自闭....

项目从开始规划到完成本文内容、到完成整个系统、再到完成系统相关文档总用时一个月，真是应了那句话，打不倒我的只会让我更强大，梦想仍未实现，我等仍需努力！挺起胸膛，撑开旗帜，刀锋向前！

# **七、参考**

* [Audio Deep Learning Made Simple: Sound Classification, Step-by-Step](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5 "Audio Deep Learning Made Simple: Sound Classification, Step-by-Step")
* [Simple audio recognition: Recognizing keywords](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb "Simple audio recognition: Recognizing keywords")
* [A Gentle Introduction to Audio Classification With Tensorflow](https://pub.towardsai.net/a-gentle-introduction-to-audio-classification-with-tensorflow-c469cb0be6f5 "A Gentle Introduction to Audio Classification With Tensorflow")
* [应用深度学习使用 Tensorflow 对音频进行分类](https://zhuanlan.zhihu.com/p/387415524 "应用深度学习使用 Tensorflow 对音频进行分类")
* SENGUPTA, NANDINI, SAHIDULLAH, MD, SAHA, GOUTAM. Lung sound classification using cepstral-based statistical features[J]. Computers in Biology and Medicine,2016,75118-129. DOI:10.1016/j.compbiomed.2016.05.013.
* [Tensorflow-Audio-Classification](https://github.com/luuil/Tensorflow-Audio-Classification "Tensorflow-Audio-Classification")

# 八、更多下载

* ****[包含所提到的数据集和代码，点击下载⏬！](https://mianbaoduo.com/o/bread/YpyYlJtt"包含所提到的数据集和代码，点击下载⏬！")****
* ****[婴儿哭声分类识别系统源码（系统完整代码），点击下载⏬！](https://mianbaoduo.com/o/bread/YpyYk5Zp"婴儿哭声分类识别系统源码（系统完整代码），点击下载⏬！")****

​
