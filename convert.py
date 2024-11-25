import numpy as np
from PIL import Image
import pretty_midi
import pandas as pd
import soundfile as sf
from IPython.display import Audio as display_audio
import os

def image_to_pitch(image_path, min_pitch=21, max_pitch=108, time_step=0.01):
    """
    将二值图像转换为音符事件
    Args:
        image_path: 图像路径
        min_pitch: 最小 MIDI pitch
        max_pitch: 最大 MIDI pitch
        time_step: 每列对应的时间步长（秒）
    Returns:
        pretty_midi 的 Note 对象列表
    """
    # 加载图像并转换为二值矩阵
    images_list = []
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    for i in range(img_array.shape[1] // 512):
        img = img_array[:, i*512:(i+1)* 512, :]
        # check 3 channels value, if all 3 channels are smaller than 128, then set it to 0, otherwise set it to 1
        img = np.all(img < 128, axis=-1)
        picture_raw = np.zeros((96, 4096))
        count = 0
        for j in range(8):
            for k in range(8):
                picture_raw[:, count * 64:(count + 1) * 64] = img[j * 96:(j + 1) * 96, k * 64:(k + 1) * 64]
                count += 1
        # reove top 4 and bottom 4 rows
        picture_raw = picture_raw[4:-4, :]
        images_list.append(picture_raw)

    images_list = [images_list[0]]

    for piano_roll in images_list:
        num_pitches, num_time_steps = piano_roll.shape
        notes = []

        for pitch_idx in range(num_pitches):
            midi_pitch = pitch_idx + 21

            # 获取当前音高的所有时间步状态
            pitch_sequence = piano_roll[pitch_idx]

            # 找出所有音符的起始和结束位置
            note_starts = np.where(np.diff(np.concatenate(([0], pitch_sequence))) == 1)[0]
            note_ends = np.where(np.diff(np.concatenate((pitch_sequence, [0]))) == -1)[0]

            # 为每个音符创建字典
            for start, end in zip(note_starts, note_ends):
                note = {
                    'pitch': int(midi_pitch),
                    'start_time': float(start * time_step),
                    'end_time': float(end * time_step),
                    'duration': float((end - start) * time_step),
                    'step': 0.0  # 初始化step，稍后更新
                }
                notes.append(note)

            # 按开始时间排序
            notes.sort(key=lambda x: x['start_time'])

            # 计算每个音符的step（与前一个音符的时间间隔）
            for i in range(len(notes)):
                if i == 0:
                    notes[i]['step'] = 0.0  # 第一个音符的step为0
                else:
                    notes[i]['step'] = round(notes[i]['start_time'] - notes[i - 1]['start_time'], 6)

            return pd.DataFrame(notes)


def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100):  # note loudness
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0

    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)

    return pm


def midi2audio(pm):
    _SAMPLING_RATE = 16000
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    # Save the audio to a file
    sf.write("output.wav", waveform, _SAMPLING_RATE)



if __name__ == "__main__":
    # 输入图像文件路径和输出 MIDI 文件夹
    image_path = "sample_500.png"  # 假设图像文件名为 samples_500.png
    output_dir = "midi_files"
    os.makedirs(output_dir, exist_ok=True)

    # 输出 MIDI 文件路径
    output_midi_path = os.path.join(output_dir, f"{os.path.splitext('samples_500_128.png')[0]}.midi")
    raw_notes = image_to_pitch(image_path)
    midi = notes_to_midi(raw_notes, output_midi_path, instrument_name="Acoustic Grand Piano")
    # save midi file
    midi.write(output_midi_path)


