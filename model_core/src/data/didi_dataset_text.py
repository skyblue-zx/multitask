import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class DiDiDatasetText(Dataset):
    def __init__(self, data, vocabulary_dict, predict=False):
        """
        构造函数
        :param data_list:
        :param audio_length:
        :param vocabulary_dict:
        """
        # 样本列表
        self.data = data
        self.predict = predict
        # 音频实际长度列表
        # 词典
        self.vocabulary_dict = vocabulary_dict

        self.label_list, self.asr_label_list, self.asr_list = DiDiDatasetText.read_data(data, self.predict)

    def __len__(self):
        """
        获取数据集长度
        :return:
        """
        return len(self.label_list)

    def __getitem__(self, index):
        """
        数据集迭代函数
        :param index:
        :return:
        """
        # 获取当前条样本
        label = self.label_list[index]
        asr_label = self.asr_label_list[index]
        asr_text = self.asr_list[index]

        # 文本数据处理
        # Todo
        # 学习re.split、re.sub等函数的使用
        text_words = [x.lower() for x in re.split(' +', re.sub('[\.,\?\!]', ' ', asr_text))]
        text_input = torch.LongTensor([int(self.vocabulary_dict.get(x, '-1')) for x in text_words if len(x) > 0])
        # Here we use the 0 to represent the padding tokens
        text_input = text_input + 1
        text_length = text_input.size(0)

        sample = {
            'label': label,
            'asr_label': asr_label,

            'text_input': text_input,
            'text_length': text_length
        }

        return sample

    @classmethod
    def read_data(cls, data, predict):
        """

        :param data:
        :param predict
        :return:
        """
        label_list = list()
        asr_label_list = list()
        asr_list = list()

        if isinstance(data, list):
            label_list.append(data[0])
            asr_label_list.append(data[1])
            asr_list.append(data[3])
        else:
            with open(data, 'r', encoding='utf-8') as f:
                for line in f:
                    # line = line.strip('\r\n').split('\t')
                    line = line.strip('\r\n').split(',')

                    if len(line) != 4:
                        continue

                    if not predict:
                        label = int(line[0].strip())
                        asr_label = int(line[1].strip()) if line[1].strip() != '' else 0

                    asr = line[3]

                    if not predict:
                        label_list.append(label)
                        asr_label_list.append(asr_label)

                    asr_list.append(asr)

        return label_list, asr_label_list, asr_list

    @classmethod
    def collate(cls, sample_list):
        """
        自定义Dataloader迭代函数
        :param sample_list:
        :return:
        """
        # 获取文本数据并重组为列表
        # x['text_input']均为Tensor
        # batch_text均为元素为Tensor的List
        batch_text = [x['text_input'] for x in sample_list]

        # 生成Pad后的Tensor
        # 元素为Tensor的List转化为一个Tensor
        # 按最大长度的Tensor作为Pad的长度
        batch_text = pad_sequence(batch_text, batch_first=True)

        # 获取并生成音频和文本实际长度的Tensor
        text_length = torch.LongTensor([x['text_length'] for x in sample_list])

        # 获取并生成Label的Tensor
        batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
        batch_asr_label = torch.tensor([x['asr_label'] for x in sample_list], dtype=torch.long)

        return (batch_text, text_length), batch_label, batch_asr_label
