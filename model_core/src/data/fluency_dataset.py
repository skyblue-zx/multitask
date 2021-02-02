import os
import torch
from pydub import AudioSegment
from torch.utils.data import Dataset

from transformers import BertTokenizer

from model_core.src.utils import media
from model_core.src.data import vggish_input
from model_core.src.models.vggish import VGGish


class FluencyDataset(Dataset):
    def __init__(self, task_type, data, audio_dir, max_seq_len, asr_pretrain_model=None, audio_pretrain_model=None,
                 predict=False, cache=True, temp_dir=None):
        """
        构造函数
        :param task_type:
        :param data:
        :param audio_dir:
        :param max_seq_len:
        :param asr_pretrain_model:
        :param audio_pretrain_model:
        :param predict:
        :param cache:
        :param temp_dir:
        :param use_cuda:
        :param master_gpu_id:
        :param gpu_id_list:
        """
        ################
        #  基础参数配置  #
        ################
        self.task_type = task_type
        # 设置样本文件路径
        self.data = data
        # 设置音频文件目录
        self.audio_dir = audio_dir
        # 设置ASR文本序列最大长度
        self.max_seq_len = int(max_seq_len)
        # 设置ASR预训练模型
        self.asr_pretrain_model = asr_pretrain_model
        # 设置音频预训练模型
        self.audio_pretrain_model = audio_pretrain_model
        # 是否为预测数据集
        self.predict = predict
        # 是否读取缓存音频
        self.cache = cache

        self.temp_dir = temp_dir

        # 创建辅助目录
        if isinstance(self.data, str):
            self.temp_dir = os.path.join(os.path.dirname(self.data), '.temp', 'wav') if self.temp_dir is None else self.temp_dir
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
        else:
            if self.temp_dir is None:
                raise RuntimeError("[ERROR] The temp dir doesn't exist.")

        ################
        #  文本部分配置  #
        ################
        if self.asr_pretrain_model is not None:
            # 配置文本分词器
            self.tokenizer = BertTokenizer.from_pretrained(self.asr_pretrain_model)
            # 设置PAD字符
            self.padding = self.tokenizer.vocab["[PAD]"]

        ################
        #  音频部分配置  #
        ################
        if self.audio_pretrain_model is not None:
            self.vggish_model = VGGish()
            pytorch_model_state = self.vggish_model.state_dict()
            checkpoint = torch.load(self.audio_pretrain_model)
            state_dict = {k: v for k, v in checkpoint.items() if k in pytorch_model_state.keys()}
            pytorch_model_state.update(state_dict)
            self.vggish_model.load_state_dict(pytorch_model_state)

        # 判断样本合法性
        self._check_config()
        # 读取样本数据
        self.label_list, self.asr_label_list, self.audio_path_list, self.asr_list = FluencyDataset.read_data(self.data,
                                                                                                             self.audio_dir,
                                                                                                             self.predict)

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
        # 获取对应索引的标签、ASRb标签、音频路径和ASR文本
        label = self.label_list[item]
        asr_label = self.asr_label_list[item]
        audio_path = self.audio_path_list[item]
        asr = self.asr_list[item]

        if self.task_type == 'single_model_audio':
            audio, _ = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            if not self.predict:
                sample = {
                    'label': label,
                    'audio': audio
                }
            else:
                sample = {
                    'audio': audio
                }

        elif self.task_type in ('single_model_audio_lstm'):
            audio, audio_length = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir,
                                                                   self.cache)

            audio = audio.squeeze(0)
            audio = audio.contiguous()

            if not self.predict:
                sample = {
                    'label': label,
                    'audio': audio,
                    'audio_length': audio_length,
                }
            else:
                sample = {
                    'audio': audio,
                    'audio_length': audio_length,
                }
        elif self.task_type in ('single_model_audio_lstm_text_gate_merge'):
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)

            audio, audio_length = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir,
                                                                   self.cache)

            audio = audio.squeeze(0)
            audio = audio.contiguous()

            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'audio': audio,
                    'audio_length': audio_length,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask
                }
            else:
                sample = {
                    'audio': audio,
                    'audio_length': audio_length,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'single_model_audio_gate_merge':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)
            audio, _ = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)
            # audio = FluencyDataset.get_audio_raw_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }
            else:
                sample = {
                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'single_model_text_embedding':
            # sample = {
            #     'label': label,
            #     'asr_label': asr_label,
            #
            #     'tokens': tokens,
            #     'segment_ids': segment_ids,
            #     'attention_mask': attention_mask,
            # }
            sample = None

        elif self.task_type == 'single_model_text_bert':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)
            if not self.predict:
                sample = {
                    'label': label,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }
            else:
                sample = {
                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'single_model_text_gate_merge':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)
            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }
            else:
                sample = {
                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'multimodel_embedding_fuse_text_bert':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)

            audio, _ = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }
            else:
                sample = {
                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'multimodel_feature_fuse_text_bert_gate_merge':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)

            audio, _ = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }
            else:
                sample = {
                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'multimodel_feature_fuse_text_embedding':
            sample = None

        elif self.task_type == 'multimodel_decision_fuse_text_embedding':
            sample = None

        elif self.task_type == 'multimodel_decision_fuse_text_bert':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)
            audio, _ = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            sample = {
                'label': label,
                'asr_label': asr_label,

                'audio': audio,

                'tokens': tokens,
                'segment_ids': segment_ids,
                'attention_mask': attention_mask,
            }

        elif self.task_type == 'multimodel_hybrid':
            tokens, segment_ids, attention_mask, _ = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)
            audio, _ = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)
            # audio = FluencyDataset.get_audio_raw_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }
            else:
                sample = {
                    'audio': audio,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                }

        elif self.task_type == 'multimodel_didi_pretrain':
            tokens, segment_ids, attention_mask, tokens_length = FluencyDataset.get_text_bert_feature(asr,
                                                                                       self.tokenizer,
                                                                                       self.max_seq_len,
                                                                                       self.padding)
            audio, audio_length = FluencyDataset.get_audio_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)
            # audio = FluencyDataset.get_audio_raw_feature(audio_path, self.vggish_model, self.temp_dir, self.cache)

            audio = audio.squeeze(0)
            audio = audio.contiguous()

            if not self.predict:
                sample = {
                    'label': label,
                    'asr_label': asr_label,

                    'audio': audio,
                    'audio_length': audio_length,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                    'tokens_length': tokens_length
                }
            else:
                sample = {
                    'audio': audio,
                    'audio_length': audio_length,

                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'attention_mask': attention_mask,
                    'tokens_length': tokens_length
                }

        else:
            sample = None
        
        return sample

    def _check_config(self):
        """
        检查数据集配置正确性
        :return:
        """
        if isinstance(self.data, list):
            if len(self.data) != 2:
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

    @classmethod
    def get_audio_feature(cls, audio_path, audio_pretrain_model, temp_dir, cache=True):
        """
        音频特征处理
        :param audio_path:
        :param audio_pretrain_model:
        :param temp_dir:
        :param cache:
        :return:
        """
        # 获取音频名称和类型
        audio_name = os.path.basename(audio_path)
        audio_type = audio_path[audio_path.rfind('.') + 1:]
        # 基于媒体文件的类型读入音频数据
        if audio_type == 'mp3':
            # MP3格式
            wav_file_path = os.path.join(temp_dir, audio_name[:audio_name.rfind('.')] + '.wav')
            if cache and os.path.exists(wav_file_path):
                sample_rate, wav_data = media.get_wav_data(wav_file_path)
            else:
                audio = AudioSegment.from_mp3(audio_path)
                audio.export(wav_file_path, format='wav')
                sample_rate, wav_data = media.get_wav_data(wav_file_path)

            # Todo

        elif audio_type == 'wav':
            # WAV格式
            sample_rate, wav_data = media.get_wav_data(audio_path)

            # Todo
        else:
            # 其他格式暂不支持
            # Todo
            pass

        # Produce a batch of log mel spectrogram examples.
        wav_data = wav_data / 32768.0
        audio = vggish_input.waveform_to_examples(wav_data, sample_rate)
        audio = torch.from_numpy(audio).unsqueeze(dim=1)
        audio = audio.float()
        audio = torch.as_tensor(audio)

        # 获得模型输出
        audio = audio_pretrain_model(audio)
        # 截取60秒音频
        audio = audio[:60 if audio.shape[0] > 60 else audio.shape[0], :]
        # 打平音频多维向量
        audio = audio.view(audio.size(0) * audio.size(1))
        # 计算pad个数
        pad_size = 737280 - audio.size(0)
        # 对不足60秒的音频进行pad
        audio = torch.cat([audio, torch.zeros(pad_size)])
        # 恢复音频多维向量
        audio = audio.view(-1, 512 * 24)
        # 增加维度生成符合Batch的数据格式
        audio = audio.unsqueeze(dim=0)

        # Todo
        # 该句代码的必要性检查
        # 提取数据加载至CPU并解析为NumPy格式
        # audio = audio.detach().cpu().numpy()
        audio = audio.detach()

        return audio, audio.shape[1]

    @classmethod
    def get_audio_raw_feature(cls, audio_path, audio_pretrain_model, temp_dir, cache=True):
        """
        音频特征处理
        :param audio_path:
        :param audio_pretrain_model:
        :param temp_dir:
        :param cache:
        :return:
        """
        # 获取音频名称和类型
        audio_name = os.path.basename(audio_path)
        audio_type = audio_path[audio_path.rfind('.') + 1:]
        # 基于媒体文件的类型读入音频数据
        if audio_type == 'mp3':
            # MP3格式
            wav_file_path = os.path.join(temp_dir, audio_name[:audio_name.rfind('.')] + '.wav')
            if cache and os.path.exists(wav_file_path):
                sample_rate, wav_data = media.get_wav_data(wav_file_path)
            else:
                audio = AudioSegment.from_mp3(audio_path)
                audio.export(wav_file_path, format='wav')
                sample_rate, wav_data = media.get_wav_data(wav_file_path)

            # Todo

        elif audio_type == 'wav':
            # WAV格式
            sample_rate, wav_data = media.get_wav_data(audio_path)

            # Todo
        else:
            # 其他格式暂不支持
            # Todo
            pass

        # Produce a batch of log mel spectrogram examples.
        wav_data = wav_data / 32768.0
        audio = vggish_input.waveform_to_examples(wav_data, sample_rate)
        audio = torch.from_numpy(audio).unsqueeze(dim=1)
        audio = audio.float()
        audio = torch.as_tensor(audio)

        return audio

    @classmethod
    def get_text_embedding_feature(cls):
        pass

    @classmethod
    def get_text_bert_feature(cls, asr, tokenizer, max_seq_len, padding):
        """
        基于Bert的文本处理
        :param asr:
        :param tokenizer:
        :param max_seq_len:
        :param padding:
        :return:
        """
        # 分词
        asr_tokens = tokenizer.tokenize(asr)
        # 根据设置的序列最大长度截取问题和回答的序列长度
        asr_tokens = asr_tokens[:int(max_seq_len) - 2]
        # 拼接问题和回答序列并增加标识符
        tokens = ["[CLS]"] + asr_tokens + ["[SEP]"]
        # 将分词序列转换为ID序列
        tokens = tokenizer.convert_tokens_to_ids(tokens)

        tokens_size = len(tokens) - 2

        # 设置token_type_ids
        segment_ids = [0] * len(tokens)
        segment_ids.extend([0] * (max_seq_len - len(segment_ids)))
        segment_ids = torch.as_tensor(segment_ids)

        # 设置attention_mask
        attention_mask = [1] * len(tokens)
        attention_mask.extend([0] * (max_seq_len - len(attention_mask)))
        attention_mask = torch.as_tensor(attention_mask)

        # 根据设置的序列最大长度补全ID序列
        tokens.extend([padding] * (max_seq_len - len(tokens)))
        tokens = torch.as_tensor(tokens)

        return tokens, segment_ids, attention_mask, tokens_size

    @staticmethod
    def read_data(data, audio_dir, predict=False):
        """
        读取样本数据
        读取原则：
        1、文本数据一次性读入；
        2、音视频和图片一次读入文件路径，字节数据在使用时再读入。
        :param data:
        :param audio_dir:
        :param predict
        :return:
        """
        label_list = list()
        asr_label_list = list()
        audio_path_list = list()
        asr_list = list()

        if isinstance(data, list):
            label_list.append(data[0])
            asr_label_list.append(data[1])
            audio_path_list.append(data[2])
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

                    audio_path = os.path.join(audio_dir, line[2])
                    asr = line[3]

                    if os.path.exists(audio_path) and asr != '':
                        if not predict:
                            label_list.append(label)
                            asr_label_list.append(asr_label)

                        audio_path_list.append(audio_path)
                        asr_list.append(asr)

        return label_list, asr_label_list, audio_path_list, asr_list
