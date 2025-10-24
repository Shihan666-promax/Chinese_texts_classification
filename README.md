# Chinese_texts_classification
A comprehensive NLP project for Chinese text classification using machine learning algorithms. This project implements a complete pipeline from text preprocessing to model evaluation.
# 中文文本分类实验

## 实验简介
本项目是基于机器学习的中文文本分类实验，使用搜狗实验室的中文文本分类数据集，实现了从文本预处理到模型训练的完整流程。

## 实验目的
- 理解文本分类任务的基本流程和关键环节
- 掌握文本的向量空间模型表示方法（如词袋模型、TF-IDF）
- 学会使用经典的机器学习分类模型实现中文文本的自动分类

## 实验任务
- 使用提供的文本分类数据集，结合分词代码对训练集和测试集进行文本预处理
- 实现文本向量化模块，使用TF-IDF将文本表示为特征向量
- 实现朴素贝叶斯和SVM分类算法，利用训练集进行模型训练
- 使用测试集评估分类模型的性能，计算准确率、F1-score等评估指标

## 数据集
- **数据来源**：搜狗实验室中文文本分类语料
- **训练集**：9个类别，每个类别1989篇文本
- **测试集**：3581篇文本
- **类别**：财经、IT、健康、体育、旅游、教育、招聘、文化、军事

## 项目结构
experiment3/
├── data/ # 原始数据
│ ├── Training Dataset/ # 训练集
│ ├── Test Dataset/ # 测试集
│ └── SegDict.TXT # 分词词典
├── output/ # 处理后的数据
│ ├── Processed_Training_Dataset/
│ └── Processed_Test_Dataset/
├── src/ # 源代码
│ ├── preprocess.py # 数据预处理
│ ├── text_classification.py # 文本分类
│ └── utils.py # 工具函数
├── saved_models/ # 训练好的模型
├── results/ # 实验结果
└── README.md
