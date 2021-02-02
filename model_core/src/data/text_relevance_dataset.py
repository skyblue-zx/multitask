import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextRelevanceDataset(Dataset):
    def __init__(self, data, max_seq_len, model_dir):
        """
        构造函数
        :param data_path:
        :param max_seq_len:
        :param model_dir:
        """
        self.data = data
        self.max_seq_len = int(max_seq_len)
        self.model_dir = model_dir

        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        self.padding = self.tokenizer.vocab["[PAD]"]

        self._check_config()

        self.label_list, self.question_list, self.answer_list = TextRelevanceDataset.read_data(self.data)

    def __len__(self):
        """
        实现数据集长度获取函数
        :return:
        """
        return len(self.label_list)

    def __getitem__(self, item):
        """
        实现数据集元素获取函数
        :param item:
        :return:
        """
        # 获取对应索引的标签、问题和答案
        label = self.label_list[item]
        question = self.question_list[item]
        answer = self.answer_list[item]

        # print(question)

        # 分词
        question_tokens = self.tokenizer.tokenize(question)
        answer_tokens = self.tokenizer.tokenize(answer)

        # print(question_tokens)

        # 根据设置的序列最大长度截取问题和回答的序列长度
        question_tokens = question_tokens[:int(self.max_seq_len / 2) - 1]
        answer_tokens = answer_tokens[:(self.max_seq_len - 3 - len(question_tokens))]

        # 拼接问题和回答序列并增加标识符
        tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + answer_tokens + ["[SEP]"]

        # 将分词序列转换为ID序列
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # 设置token_type_ids
        segment_ids = [0] * len(tokens)
        segment_ids.extend([0] * (self.max_seq_len - len(segment_ids)))
        segment_ids = torch.as_tensor(segment_ids)

        # 设置attention_mask
        attention_mask = [1] * len(tokens)
        attention_mask.extend([0] * (self.max_seq_len - len(attention_mask)))
        attention_mask = torch.as_tensor(attention_mask)

        # 根据设置的序列最大长度补全ID序列
        tokens.extend([self.padding] * (self.max_seq_len - len(tokens)))
        tokens = torch.as_tensor(tokens)

        sample = {
            'tokens': tokens,
            'segment_ids': segment_ids,
            'attention_mask': attention_mask,
            'label': label
        }

        return sample

    def _check_config(self):
        """
        检查数据集配置正确性
        :return:
        """
        if isinstance(self.data, list):
            if len(self.data) != 3:
                raise RuntimeError("Data format is error." + self.data)
        else:
            if self.data:
                if not os.path.exists(self.data):
                    raise RuntimeError("Data do not exist at " + self.data)

        # if self.model_dir:
        #     if not os.path.exists(self.model_dir):
        #         raise RuntimeError("Model dir do not exist at" + self.model_dir)

        return

    def get_labels_num(self):
        """
        获取label个数
        :return:
        """
        return len(set(self.label_list))

    @staticmethod
    def read_data(data):
        """
        读取样本数据
        :param data_path:
        :return:
        """
        label_list = list()
        question_list = list()
        answer_list = list()

        if isinstance(data, list):
            label_list.append(data[0])
            question_list.append(data[1])
            answer_list.append(data[2])
        else:
            with open(data, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\r\n').split('\t')

                    if len(line) != 3:
                        continue

                    label = int(line[0])
                    question = line[1]
                    answer = line[2]

                    if question != '' and answer_list != '':
                        label_list.append(label)
                        question_list.append(question)
                        answer_list.append(answer)

        return label_list, question_list, answer_list


# 仅供单元测试
if __name__ == '__main__':
    dataset = TextRelevanceDataset('/Users/zhaixiao/workplace/python/tal_model_train_and_predict_env/model_core/_instance_main/sample.txt',
                                   1000,
                                   'bert-base-chinese')

    for sample in dataset:
        print(sample)
        break
