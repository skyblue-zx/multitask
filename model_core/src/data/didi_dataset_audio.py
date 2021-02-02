import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from pyAudioAnalysis.audioBasicIO import read_audio_file
from pyAudioAnalysis.ShortTermFeatures import feature_extraction


class DiDiDatasetAudio(Dataset):
    def __init__(self, data, audio_dir, audio_length=None, predict=False):
        """
        构造函数
        :param data_list:
        :param audio_length:
        :param vocabulary_dict:
        """
        # 样本列表
        self.data = data
        self.audio_dir = audio_dir
        self.predict = predict
        # 音频实际长度列表
        self.audio_length = audio_length

        self.label_list, self.asr_label_list, self.audio_list = DiDiDatasetAudio.read_data(data,
                                                                                           self.audio_dir,
                                                                                           self.predict)

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
        audio_path = self.audio_list[index]

        # 音频数据处理
        audio_name = os.path.basename(audio_path)
        [Fs, x] = read_audio_file(audio_path)
        audio_feature = feature_extraction(x, Fs, 0.020 * Fs, 0.01 * Fs, False)[0].T
        audio_feature = audio_feature.astype('float32')
        audio_input = torch.FloatTensor(audio_feature)
        if self.audio_length is not None:
            audio_input = audio_input[:self.audio_length, :]
        audio_length = audio_input.size(0)

        sample = {
            'label': label,
            'asr_label': asr_label,

            'audio_name': audio_name,
            'audio_input': audio_input,
            'audio_length': audio_length
        }

        return sample

    @classmethod
    def read_data(cls, data, audio_dir, predict):
        """

        :param data:
        :param audio_dir:
        :param predict
        :return:
        """
        label_list = list()
        asr_label_list = list()
        audio_path_list = list()

        if isinstance(data, list):
            label_list.append(data[0])
            asr_label_list.append(data[1])
            audio_path_list.append(data[2])
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

                    audio_path = os.path.join(audio_dir, line[2])

                    if os.path.exists(audio_path):
                        if not predict:
                            label_list.append(label)
                            asr_label_list.append(asr_label)

                        audio_path_list.append(audio_path)

        return label_list, asr_label_list, audio_path_list

    @classmethod
    def collate(cls, sample_list):
        """
        自定义Dataloader迭代函数
        :param sample_list:
        :return:
        """
        # 获取音频和文本数据并重组为列表
        # x['audio_input']和x['text_input']均为Tensor
        # batch_audio和batch_text均为元素为Tensor的List
        batch_audio = [x['audio_input'] for x in sample_list]

        # 生成Pad后的Tensor
        # 元素为Tensor的List转化为一个Tensor
        # 按最大长度的Tensor作为Pad的长度
        batch_audio = pad_sequence(batch_audio, batch_first=True)

        # 获取并生成音频和文本实际长度的Tensor
        audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])

        # 获取并生成Label的Tensor
        batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
        batch_asr_label = torch.tensor([x['asr_label'] for x in sample_list], dtype=torch.long)
        # 获取并生成音频名称的Tensor
        batch_name = [x['audio_name'] for x in sample_list]

        return (batch_audio, audio_length), batch_label, batch_asr_label, batch_name
