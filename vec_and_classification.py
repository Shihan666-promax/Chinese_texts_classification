# 中文文本分类实验 - 文本向量化与分类模型
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
class TextClassifier:
    def __init__(self, train_data_dir, test_data_dir):
        #初始化文本分类器
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.vectorizer = None
        self.models = {}
        self.results = {}
        self.category_mapping = {
            "财经": 0,
            "IT": 1,
            "健康": 2,
            "体育": 3,
            "旅游": 4,
            "教育": 5,
            "招聘": 6,
            "文化": 7,
            "军事": 8
        }
        
        self.reverse_category_mapping = {v: k for k, v in self.category_mapping.items()}
    
    def load_data(self, data_dir, is_train=True):
        #加载训练集或测试集数据
        texts = []
        labels = []
        file_count = 0
        
        if is_train:
            # 训练集：按类别目录加载
            for category_name, label_id in self.category_mapping.items():
                category_dir = os.path.join(data_dir, category_name)
                if os.path.exists(category_dir):
                    for filename in os.listdir(category_dir):
                        if filename.endswith('.txt'):
                            file_path = os.path.join(category_dir, filename)
        else:
            for filename in os.listdir(data_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(data_dir, filename)
        return texts, labels
    
    def create_tfidf_vectorizer(self, max_features=10000, min_df=2, max_df=0.8):
        #创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2), 
            sublinear_tf=True    
        )
        return self.vectorizer
    
    def train_models(self, X_train, y_train):
        # 朴素贝叶斯模型
        nb_model = MultinomialNB(alpha=0.1)
        nb_model.fit(X_train, y_train)
        self.models['NaiveBayes'] = nb_model
        
        # 支持向量机模型
        svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        results = {}
        for model_name, model in self.models.items():
            # 预测
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'y_pred': y_pred,
                'classification_report': classification_report(y_test, y_pred, target_names=[self.reverse_category_mapping[i] for i in sorted(self.category_mapping.values())], output_dict=True)
            }
            
            # 打印结果
            print(f"准确率: {accuracy:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(classification_report(y_test, y_pred, target_names=[self.reverse_category_mapping[i] for i in sorted(self.category_mapping.values())]))
        
        self.results = results
        return results
    
    def save_models(self, save_dir):
        
        #保存
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.vectorizer:
            joblib.dump(self.vectorizer, os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f'{model_name}_model.pkl'))
        results_df = self._results_to_dataframe()
        results_df.to_csv(os.path.join(save_dir, 'model_results.csv'), index=False, encoding='utf-8')
    
    def _results_to_dataframe(self):
        #将结果转换为DataFrame
        data = []
        for model_name, result in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'F1_Score': result['f1_score'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std'],
                'Prediction_Time': result['prediction_time']
            })
        return pd.DataFrame(data)
    
    def run_complete_pipeline(self, max_features=8000):
        train_texts, train_labels = self.load_data(self.train_data_dir, is_train=True)
        test_texts, test_labels = self.load_data(self.test_data_dir, is_train=False)
        self.create_tfidf_vectorizer(max_features=max_features)
        # 训练集向量化
        X_train = self.vectorizer.fit_transform(train_texts)
        y_train = np.array(train_labels)
        # 测试集向量化
        X_test = self.vectorizer.transform(test_texts)
        print(f"特征维度: {X_train.shape[1]}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        self.train_models(X_train, y_train)
        # 为了评估，我们将训练集分成训练和验证集
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        eval_models = {}
        for model_name, model in self.models.items():
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train_split, y_train_split)
            eval_models[model_name] = model_clone
        
        # 评估模型
        self.evaluate_models(X_val_split, y_val_split)
        # 6. 保存模型
        save_dir = os.path.join(os.path.dirname(self.train_data_dir), 'saved_models')
        self.save_models(save_dir)
        
        results_df = self._results_to_dataframe()
        print("\n模型性能排名:")
        results_sorted = results_df.sort_values('Accuracy', ascending=False)
        print(results_sorted.to_string(index=False))
        
        best_model_info = results_sorted.iloc[0]
        print(f"\n最佳模型: {best_model_info['Model']}")
        print(f"最佳准确率: {best_model_info['Accuracy']:.4f}")
        print(f"最佳F1-score: {best_model_info['F1_Score']:.4f}")
        
        return self.results

def main():
    #主函数
    train_data_dir = r"E:\nlp\experiment3_data\experiment3_data\output\Processed_Training_Dataset"
    test_data_dir = r"E:\nlp\experiment3_data\experiment3_data\output\Processed_Test_Dataset"
    classifier = TextClassifier(train_data_dir, test_data_dir)
    results = classifier.run_complete_pipeline(max_features=8000)
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_report = results[best_model_name]['classification_report']
    # 转换为DataFrame以便更好显示
    report_df = pd.DataFrame(best_report).transpose()
    print(report_df.round(4).to_string())

if __name__ == "__main__":
    main()