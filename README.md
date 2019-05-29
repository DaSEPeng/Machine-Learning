本模块整理机器学习相关内容，涉及机器学习算法的核心想法、数学推导与代码实践   
<br>
路线图：
![Road](road.jpg)

-------------------------------------------------


# 理论内容



| Content      |    Notes | Project  |References | 
| :-------- | :--------| :----- | :----- | 
| **1. 机器学习概论**  |  |     | |
| 机器学习概述| | | | 
| **2. 线性模型**| | | |
| 感知机     |    | [Iris DataSet](http://archive.ics.uci.edu/ml/datasets/iris)   ||
| Logistic回归与Softmax回归      |     | [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)  ||
|SVM | |  [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)| |
|线性回归 | |[工业蒸汽量预测](https://tianchi.aliyun.com/competition/entrance/231693/information) ||
| **3. 神经网络**| | ||
|MLP |[PDF](/NeuralNetwork/NN.pdf) [PPT](/NeuralNetwork/NN.pptx) | [Mnist Datasets](http://yann.lecun.com/exdb/mnist/)|[Backpropagation](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)|
|CNN | | ||
|RNN | | ||
|**4. 决策树与集成学习** | | ||
| 决策树| | [Online Shoppers Purchasing Intention Dataset Data Set](http://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)|《数据挖掘导论》|
| 集成学习与随机森林 | |  [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)|[决策树与随机森林（邹博）](/RandomForest/DT_RF_zb.pdf)、[集成学习（周志华）](/RandomForest/EnsembleLearning_zhh.pdf)、《数据挖掘导论》|
| AdaBoost| | ||
| GBDT | | ||
| Xgboost| | ||
|**5. kNN** | |  [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)||
|kNN | | ||
|距离度量学习 | | ||
|**6. 贝叶斯分类器** | | ||
|朴素贝叶斯 | | ||
|贝叶斯网络 | | ||
|正态贝叶斯 | | ||
|**7. 图模型** | | | |
|HMM | | ||
|CRF | | | | 
|**8. LDA** | | ||
|LDA | | | |
|KLDA | | | |
|**9. 降维** | | |  |
|PCA | | | |
|流形学习 | | | |
|**10. 聚类** | | | |
|kMeans | | |[机器学习（吴恩达）](https://study.163.com/course/courseMain.htm?courseId=1004570029&_trace_c_p_k2_=ae72248b13a941bea4881d06148a09fd)、[K-means学习进阶以及它与EM的关系](https://blog.csdn.net/u010159842/article/details/45954961)、[机器学习（邹博）](https://www.bilibili.com/video/av23585080/?p=16) |
|EM算法 | | | |
|**11. 迁移学习** | | | | 
| 迁移学习简介|[PDF](/TransferLearning/transfer_HYL.pdf) [Video](https://www.bilibili.com/video/av35932863/?p=27)| |[Everything about Transfer Learning](http://transferlearning.xyz/) | 
|**12. 强化学习** | | | |
|蒙特卡洛算法 | | | |
|价值迭代 | | | |
|策略迭代 | | | |
|时序差分算法 | | | |
|**13. 机器学习理论** | | | |
|VC维 | | | | 

注：<br> 

- PPT版本笔记有相关注释与思考；
- 本系列笔记仅供学习阅览，如需传播或作他用，请标明出处；



------------------------------------------------



# 实践内容



|    Project    |    Target | Data  | Methods | Report | Code|
| :-------- | :--------| :----- |:-----|:-----|:----|
| Project 1<br>(update 2019/06/23)  | Classification | [Online Shoppers Purchasing Intention Dataset Data Set](http://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)    |逻辑回归、决策树、神经网络|  | |
| Project 2<br>(update 2019/06/23)  |Text Classification   | [Naive Bayes algorithm for learning to classify text](http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html)    |主题模型(pLSA、LDA、etc.)         |    |     |
| Project 3<br>(update 2019/06/23)  |  Digital Recognition |[MNIST](http://yann.lecun.com/exdb/mnist/)     |  DNN,CNN,SOTA      |    |     |
| Project 4<br>(update 2019/06/23)  | Face Recognition & Expression Clustering |  [CMU Machine Learning Faces](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html) |     |    |     |





--------------------------------------------------

# 参考资料

**机器学习入门**
- [Google: Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)

**机器学习基础**
- [机器学习](http://58.198.176.86/qwdong/machinelearning/)，董启文老师讲授  
  - 《统计学习方法（第二版）》，李航著 &emsp; [[Code实现](https://github.com/Dod-o/Statistical-Learning-Method_Code)]&emsp;[[知乎参考读物](https://zhuanlan.zhihu.com/p/36378498)]       
  - 《机器学习》， 周志华著    
- [机器学习](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html)， 李宏毅 &emsp; [[Video](https://www.bilibili.com/video/av10590361?from=search&seid=3689001450384077781)]   
  - 《模式识别与机器学习》， 马春鹏译 
  - PRML相关Slides： [FUDAN PRML](https://zfhu.ac.cn/PRML-Spring19-Fudan/)&emsp;[DaSE PRML](https://github.com/ECNUdase/Seminar-PRML)
- [CS229N](http://cs229.stanford.edu/syllabus.html)，吴恩达
- [机器学习, 白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)

**机器学习实战**
- [慕课网：Python3玩转机器学习](https://coding.imooc.com/class/169.html)
- [邹博的机器学习课程](https://www.bilibili.com/video/av23585080?from=search&seid=162992797617774420)
- [kaggle案例实战班](https://www.bilibili.com/video/av42834527?from=search&seid=7750784603776284569)
- [Python机器学习Kaggle案例实战](https://www.bilibili.com/video/av35531698?from=search&seid=7750784603776284569)

**机器学习的数学理论**
- 《Machine Learning：A Probabilistic Perspective》 
- 《The Elements of Statistical Learning》

**附加阅读**
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/index.html)




------------------------------------------------------------

<div style="text-align:center;">
This page maintained by @ <a href="https://simplelp.github.io/">SimplePeng</a>, 	
<a href="https://github.com/SimpleLP/Machine-Learning/">View on GitHub</a>
</div>
