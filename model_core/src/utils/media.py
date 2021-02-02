import wave
import numpy as np


def get_wav_data(wav_file_path):
    """
    读取WAV文件中的音频数据
    :param wav_file_path:
    :return:
    """
    # 打开WAV格式的音频文件
    wav_file = wave.open(wav_file_path, 'rb')

    # 获取音频格式
    # 获取音频通道的数量（1对于单声道，2对于立体声）
    nchannels = wav_file.getnchannels()
    # 获取以字节为单位的样本宽度
    sample_width = wav_file.getsampwidth()
    # 获取采样频率
    framerate = wav_file.getframerate()
    # 获取音频帧数
    numframes = wav_file.getnframes()

    # print("channel: ", nchannels)
    # print("sample_width: ", sample_width)
    # print("framerate: ", framerate)
    # print("numframes: ", numframes)

    # 读取音频数据
    # 暂仅支持单通道双字节格式的音频数据
    if nchannels == 1:
        buffer_data = wav_file.readframes(numframes)
        if sample_width == 2:
            wav_data = np.frombuffer(buffer_data, dtype=np.int16)
        else:
            raise Exception("[ERROR] sample_width is not 2.")
    else:
        raise Exception("[ERROR] nchannels is not 1.")

    return framerate, wav_data

    # # 存储音频数据为WAV文件
    # audiosegment = AudioSegment(data=memoryview(wav_data[1315008:]), sample_width=sample_width, frame_rate=framerate, channels=nchannels)
    # audiosegment.export('/Users/zhaixiao/test1.wav', format='wav')