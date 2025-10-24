
# 中文文本分类实验 - 数据预处理模块
import os
import chardet
from collections import Counter

class ChineseSegmenter:
    def __init__(self, dict_path, stopwords_path):
        # 初始化
        self.dictionary = self.load_dictionary(dict_path)
        self.stopwords = self.load_stopwords(stopwords_path)
        self.max_word_len = max(len(word) for word in self.dictionary) if self.dictionary else 0
    
    def load_dictionary(self, dict_path):
        # 分词词典
        dictionary = set()
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        dictionary.add(word)
            return dictionary
        except FileNotFoundError:
            return set()
        except Exception as e:
            return set()
    def load_stopwords(self, stopwords_path):
        # 加载停用词词典
        stopwords = set()
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
            
            return stopwords
        except FileNotFoundError:
            
            return set()
        except Exception as e:
            return set()
    def forward_max_match(self, text, max_len=None):
        #正向最大匹配算法
        if max_len is None:
            max_len = self.max_word_len
        
        words = []  # 存储分词结果
        index = 0   # 当前处理位置指针
        text_len = len(text)
        
        # 遍历整个文本
        while index < text_len:
            matched = False  # 标记是否匹配到词
            
            # 从最大长度开始尝试匹配，逐渐减小长度
            for length in range(min(max_len, text_len - index), 0, -1):
                word = text[index:index + length]  # 截取候选词
                
                # 如果候选词在词典中，则匹配成功
                if word in self.dictionary:
                    words.append(word)    # 将词加入结果列表
                    index += length       # 移动指针到下一个位置
                    matched = True        # 标记已匹配
                    break                 # 跳出当前循环，处理下一个位置
            
            # 如果没有匹配到任何词，按单字切分
            if not matched:
                words.append(text[index])  # 将单字作为词加入结果
                index += 1                 # 指针前进一位
        
        return words
    
    def remove_stopwords(self, words):
        #去除停用词
        return [word for word in words if word not in self.stopwords]
    
    def segment_text(self, text, remove_stopwords=True):
        # 对文本进行分词和停用词去除
        # 分词
        segmented_words = self.forward_max_match(text)
        
        # 去除停用词
        if remove_stopwords and self.stopwords:
            segmented_words = self.remove_stopwords(segmented_words)
        
        return segmented_words

def detect_encoding(file_path):
    
    #检测文件编码
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            return encoding, confidence
    except Exception as e:
        
        return None, 0

def convert_to_utf8(input_file_path, output_file_path):
    
    #将文件转换为UTF-8编码
    
    try:
        encoding, confidence = detect_encoding(input_file_path)
        if encoding is None:
            encoding = 'gbk'  
        with open(input_file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        return False

def convert_dataset_encoding(input_dir, temp_dir, category_mapping, dataset_type="训练集"):
    #将整个数据集的编码统一转换为UTF-8
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    converted_count = 0
    if dataset_type == "训练集":
        # 训练集：按类别目录处理
        for category_code, category_name in category_mapping.items():
            category_input_dir = os.path.join(input_dir, category_code)
            category_temp_dir = os.path.join(temp_dir, category_name)
            if not os.path.exists(category_temp_dir):
                os.makedirs(category_temp_dir)
            if not os.path.exists(category_input_dir):
                continue
            file_list = [f for f in os.listdir(category_input_dir) if f.endswith('.txt')]
            for filename in file_list:
                input_file_path = os.path.join(category_input_dir, filename)
                temp_file_path = os.path.join(category_temp_dir, filename)
                if convert_to_utf8(input_file_path, temp_file_path):
                    converted_count += 1
                if converted_count % 100 == 0:
                    print(f"  已转换 {converted_count} 个文件...")
    
    else:
        # 测试集：直接处理根目录下的文件
        if not os.path.exists(input_dir):
            return temp_dir, 0
        file_list = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        
        for filename in file_list:
            input_file_path = os.path.join(input_dir, filename)
            temp_file_path = os.path.join(temp_dir, filename)
            
            if convert_to_utf8(input_file_path, temp_file_path):
                converted_count += 1
            
            if converted_count % 10 == 0:
                print(f"  已转换 {converted_count} 个文件...")
    
    
    return temp_dir, converted_count

def process_training_dataset(input_dir, output_dir, segmenter, category_mapping):
    #处理训练集
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    processed_count = 0
    total_files = 0
    
    # 首先统计总文件数
    for category_code, category_name in category_mapping.items():
        category_input_dir = os.path.join(input_dir, category_name)  # 注意：这里使用转换后的目录结构
        if os.path.exists(category_input_dir):
            files = [f for f in os.listdir(category_input_dir) if f.endswith('.txt')]
            total_files += len(files)
    # 遍历每个类别目录
    for category_code, category_name in category_mapping.items():
        category_input_dir = os.path.join(input_dir, category_name)  
        category_output_dir = os.path.join(output_dir, category_name)
        
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)
        if not os.path.exists(category_input_dir):
            continue
        # 处理该类别下的所有文本文件
        file_list = [f for f in os.listdir(category_input_dir) if f.endswith('.txt')]
        category_count = 0
        
        for filename in file_list:
            input_file_path = os.path.join(category_input_dir, filename)
            output_file_path = os.path.join(category_output_dir, filename)
            
            try:
                # 读取原始文本（现在已经是UTF-8编码）
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if not text:  # 跳过空文件
                    continue
                
                # 分词和去除停用词
                segmented_words = segmenter.segment_text(text, remove_stopwords=True)
                
                # 保存处理后的文本
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(' '.join(segmented_words))
                
                processed_count += 1
                category_count += 1
                
                # 显示进度
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count}/{total_files} 个文件 ({processed_count/total_files*100:.1f}%)")
                    
            except Exception as e:
                print(f"处理文件时出错 {input_file_path}: {e}")
        
        print(f"  类别 {category_name} 完成: {category_count} 个文件")
    
    print(f"训练集处理完成, 共处理 {processed_count} 个文件")
    return processed_count

def process_test_dataset(input_dir, output_dir, segmenter, category_mapping):
    #处理测试集
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    processed_count = 0
    
    # 测试集直接放在输入目录下
    if not os.path.exists(input_dir):
        
        return 0
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    
    for filename in file_list:
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        
        try:
            # 读取原始文本,现在已经是UTF-8编码
            with open(input_file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:  
                continue
            segmented_words = segmenter.segment_text(text, remove_stopwords=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(segmented_words))
            
            processed_count += 1
            
            
        except Exception as e:
            print(f"e")
    
    print(f"测试集处理完成 共处理 {processed_count} 个文件")
    return processed_count

def main():
    dict_path = r"E:\nlp\experiment3_data\experiment3_data\data\SegDict.TXT"  # 分词词典路径
    stopwords_path = r"E:\nlp\experiment1_data\experiment1_data\data\stoplist.txt"  # 停用词词典路径
    
    # 原始训练集和测试集路径
    original_train_input_dir = r"E:\nlp\experiment3_data\experiment3_data\data\Training Dataset"  # 原始训练集输入目录
    original_test_input_dir = r"E:\nlp\experiment3_data\experiment3_data\data\Test Dataset"       # 原始测试集输入目录
    # 输出目录
    base_output_dir = r"E:\nlp\experiment3_data\experiment3_data\output"
    # 临时目录,用于存储UTF-8编码的文件
    temp_dir = os.path.join(base_output_dir, "temp_utf8")
    train_temp_dir = os.path.join(temp_dir, "Training_Dataset_UTF8")
    test_temp_dir = os.path.join(temp_dir, "Test_Dataset_UTF8")
    # 最终输出目录
    train_output_dir = os.path.join(base_output_dir, "Processed_Training_Dataset")  # 处理后的训练集输出目录
    test_output_dir = os.path.join(base_output_dir, "Processed_Test_Dataset")       # 处理后的测试集输出目录
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    category_mapping = {
        "C000008": "财经",
        "C000010": "IT", 
        "C000013": "健康",
        "C000014": "体育",
        "C000016": "旅游",
        "C000020": "教育",
        "C000022": "招聘", 
        "C000023": "文化",
        "C000024": "军事"
    }
    
    segmenter = ChineseSegmenter(dict_path, stopwords_path)
    
    if not segmenter.dictionary:
        return
    train_temp_dir, train_converted_count = convert_dataset_encoding(
        original_train_input_dir, train_temp_dir, category_mapping, "训练集"
    )
    test_temp_dir, test_converted_count = convert_dataset_encoding(
        original_test_input_dir, test_temp_dir, category_mapping, "测试集"
    )
    train_count = process_training_dataset(train_temp_dir, train_output_dir, segmenter, category_mapping)
    test_count = process_test_dataset(test_temp_dir, test_output_dir, segmenter, category_mapping)
if __name__ == "__main__":
    main()