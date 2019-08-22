本模块整理机器学习与深度学习相关内容，涉及机器学习（统计机器学习、概率图模型、神经网络）算法的核心想法、数学推导与代码实践   
<br>

-------------------------------------------------


# 理论内容


| Content      |   References   |Notes | 
| :-------- | :-------- | :----- | 
| **1. 机器学习概论**  |  |     |
| 机器学习概述| |[PDF](/Introduction/Introduction.pdf) [PPT](/Introduction/Introduction.pptx) | 
| **2. 线性模型**| | |
|线性回归 | [回归（邹博）](https://www.bilibili.com/video/av23585080/?p=8) |[PDF](/LinearRegression/LinearRegression.pdf) [Video](https://www.bilibili.com/video/av31989606/)|
| 感知机   |   |[PDF](/Perceptron/Perceptron.pdf) [PPT](/Perceptron/Perceptron.pptx)  |
| Logistic回归   |     ||
|SVM | | |
| **3. 神经网络**| | |
|MLP |[Backpropagation](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)|[PDF](/NeuralNetwork/NN.pdf) [PPT](/NeuralNetwork/NN.pptx) |
|CNN | | |
|RNN | | |
|**4. 决策树与集成学习** | | |
| 决策树| 《数据挖掘导论》||
| 集成学习与随机森林 | [决策树与随机森林（邹博）](/RandomForest/DT_RF_zb.pdf)、[集成学习（周志华）](/RandomForest/EnsembleLearning_zhh.pdf)、《数据挖掘导论》||
| AdaBoost| 《数据挖掘导论》| |
| GBDT & Xgboost | [Xgboost---Python数据分析与机器学习实战](https://www.bilibili.com/video/av29442085/?p=7)、[Introduction to Boosted Trees (Tianqi Chen)](https://homes.cs.washington.edu/~tqchen/data/pdf/BoostedTree.pdf)、[Introduction to LightGBM (Taifeng Wang@Microsoft)](https://www.bilibili.com/video/av47496956/?spm_id_from=333.788.videocard.1)||
|**5. kNN** | |  |
|kNN | |[PDF](/kNN/kNN.pdf) [PPT](/kNN/kNN.pptx) |
|距离度量学习 | [距离度量学习论文阅读](https://zhuanlan.zhihu.com/p/26382052)| |
|**6. 贝叶斯分类器** | | |
|朴素贝叶斯 ||[PDF](/NaiveBayes/NaiveBayes.pdf) [PPT](/NaiveBayes/NaiveBayes.pptx) |
|贝叶斯网络 | | |
|正态贝叶斯 | | |
|**7. 图模型** | | | 
|HMM | | |
|CRF | |  | 
|**8. LDA** | | |
|LDA | | | 
|KLDA | | | 
|**9. 降维** | | |  
|PCA | | | 
|流形学习 | | | 
|**10. 聚类** | | | 
|kMeans | [机器学习（吴恩达）](https://study.163.com/course/courseMain.htm?courseId=1004570029&_trace_c_p_k2_=ae72248b13a941bea4881d06148a09fd)、[K-means学习进阶以及它与EM的关系](https://blog.csdn.net/u010159842/article/details/45954961)、[机器学习（邹博）](https://www.bilibili.com/video/av23585080/?p=16)| |
|EM算法 | | | 
|**11. 迁移学习** | | | 
| 迁移学习简介|[PDF](/TransferLearning/transfer_HYL.pdf) [Video](https://www.bilibili.com/video/av35932863/?p=27)、[Everything about Transfer Learning](http://transferlearning.xyz/) ||  
|**12. 强化学习** | | | 
|蒙特卡洛算法 | | | 
|价值迭代 | | | 
|策略迭代 | | |
|时序差分算法 | | | 
|**13. 机器学习理论** | | | 
|VC维 | | | 

注：<br> 

- PPT版本笔记有相关注释与思考；
- 本系列笔记仅供学习阅览，如需传播或作他用，请标明出处；



------------------------------------------------



# 实践内容



|    Project    |    Target | Data  | Methods | Report | Code|
| :-------- | :--------| :----- |:-----|:-----|:----|
| | | | | | 





--------------------------------------------------

# 参考资料

**入门**
- [Google: Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
- [Neural Networks and Deep Learning, Michael Nielsen](http://neuralnetworksanddeeplearning.com/)

**基础**
- 《机器学习》， 周志华著；《统计学习方法（第二版）》，李航著 &emsp;[[知乎参考读物](https://zhuanlan.zhihu.com/p/36378498)]  
  - [机器学习](http://58.198.176.86/qwdong/machinelearning/)，董启文老师讲授 
- 《模式识别与机器学习》， 马春鹏译
  - PRML相关Slides： [FUDAN PRML](https://zfhu.ac.cn/PRML-Spring19-Fudan/)&emsp;[DaSE PRML](https://github.com/ECNUdase/Seminar-PRML)
  - [机器学习, 白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)&emsp;[Note](https://github.com/ws13685555932/machine_learning_derivation) 
- [神经网络与深度学习](https://nndl.github.io/) ，邱锡鹏著
    - [机器学习](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html)， 李宏毅 
- 《深度学习》，Lan Goodfellow等著
    - [李宏毅MLDS](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)
    - [CMU Introduction to Deep Learning](http://deeplearning.cs.cmu.edu/)
- 《概率图模型：原理与技术》Daphne Koller著
    - [CMU 10-708: Probabilistic Graphical Models](https://sailinglab.github.io/pgm-spring-2019/)
    - [Stanford CS228: Probabilistic Graphical Model](https://cs228.stanford.edu/)

**实战**
- [慕课网：Python3玩转机器学习](https://coding.imooc.com/class/169.html)
- [邹博机器学习](https://www.bilibili.com/video/av23585080?from=search&seid=162992797617774420)
- [邹博深度学习](https://www.bilibili.com/video/av50327129?from=search&seid=1080630462980792268)

**数学理论**
- 《Machine Learning：A Probabilistic Perspective》 
- 《[An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)》
  - [Stanford Lagunita: Statistical Learning](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about)
  - [UCB: Introduction to Machine Learning](https://people.eecs.berkeley.edu/~jrs/189/)
- 《The Elements of Statistical Learning》
- 统计机器学习，机器学习导论，张志华
- 机器学习，徐亦达
- [Joan Bruna：MathsDL](https://github.com/joanbruna)



**附加资料**
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/index.html)
- [AI算法工程师笔记](http://www.huaxiaozhuan.com/)
- 机器学习（Coursera版本主要关注大规模机器学习；斯坦福版本主要关注理论推导），吴恩达
- 深度学习（Coursera版本；CS230），吴恩达
- 机器学习基础、机器学习技法，林轩田



------------------------------------------------------------

<div style="text-align:center;">
This page maintained by @ <a href="https://simplelp.github.io/">SimplePeng</a>, 	
<a href="https://github.com/SimpleLP/Machine-Learning/">View on GitHub</a>
</div>
