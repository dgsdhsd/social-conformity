# -*- codeing = utf-8 -*-
# @Time : 2025/12/4 15:37
# @Author : 星空噩梦
# @File ： saccade_calculate_all.py
# @Software : PyCharm

import os
import numpy as np
import pandas as pd


# ====== 工具函数：解析 gazeDirection 字段为 numpy 向量 ======
def parse_vec(value):
    """
    把一行的 gazeDirectionNormalizedLeft/Right 转成 numpy 向量.
    支持几种常见格式:
        [0.1, 0.2, 0.9]
        (0.1,0.2,0.9)
        0.1,0.2,0.9
    如果本来就是 list/ndarray 也直接转.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # 已经是向量
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=float)
        return arr

    # 字符串情况
    if isinstance(value, str):
        s = value.strip().replace('[', '').replace(']', '') \
                        .replace('(', '').replace(')', '')
        # 按逗号或分号拆
        parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip() != '']
        if len(parts) == 0:
            return None
        nums = [float(p) for p in parts]
        return np.array(nums, dtype=float)

    # 其他类型，尽量转一下
    try:
        return np.array([float(value)], dtype=float)
    except Exception:
        return None


def get_eye_dir(row):
    """
    对一行数据，取 left/right 眼方向向量的平均，并归一化.
    如果都缺失/为0，返回 None.
    """
    v_left = parse_vec(row.get('gazeDirectionNormalizedLeft'))
    v_right = parse_vec(row.get('gazeDirectionNormalizedRight'))

    if v_left is not None and v_right is not None and v_left.shape == v_right.shape:
        v = (v_left + v_right) / 2.0
    elif v_left is not None:
        v = v_left
    elif v_right is not None:
        v = v_right
    else:
        return None

    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    return v / norm


def mean_eye_dir(gaze_df, start_idx, end_idx):
    """
    计算在 [start_idx, end_idx] (包含 end_idx) 区间内眼睛方向的平均向量.
    """
    dirs = []
    for idx in range(start_idx, end_idx + 1):
        row = gaze_df.iloc[idx]
        v = get_eye_dir(row)
        if v is not None:
            dirs.append(v)
    if not dirs:
        return None
    v_mean = np.mean(dirs, axis=0)
    norm = np.linalg.norm(v_mean)
    if norm == 0:
        return None
    return v_mean / norm


# ====== 从 fixation 序列中提取 fixation 事件 ======
def extract_fixation_events(fix_series, start_idx, end_idx):
    """
    在 [start_idx, end_idx] (end_idx 为闭区间) 上, 从 fixation 标号中提取事件:
    连续 fixation != -1 视为一个 fixation.

    返回: [(start, end), ...] 这里 end 也是包含这个 index.
    """
    events = []
    current_id = None
    seg_start = None

    for idx in range(start_idx, end_idx + 1):
        fid = fix_series.iloc[idx]

        if fid != -1:
            # 进入某个 fixation
            if current_id is None:
                current_id = fid
                seg_start = idx
            else:
                # 已经在 fixation 中，若编号变了，则认为前一个 fixation 结束
                if fid != current_id:
                    events.append((seg_start, idx - 1))
                    current_id = fid
                    seg_start = idx
        else:
            # fid == -1，当前不在 fixation
            if current_id is not None:
                # 结束当前 fixation
                events.append((seg_start, idx - 1))
                current_id = None
                seg_start = None

    # 序列结束时还有未闭合的 fixation
    if current_id is not None and seg_start is not None:
        events.append((seg_start, end_idx))

    return events


# ====== 根据 state 划分 C 段（贴近你原来的逻辑版本） ======
def split_c_segments(state_series):
    """
    根据 state 列划分 C 段:
    规则: B -> C 开始, C -> A / over 结束.
    返回 [(start_idx, end_idx), ...] 这里 end_idx 为闭区间 (即 last-1).
    """
    segments = []
    n = len(state_series)
    current = 1  # 和你原来一样，从1开始

    while current < n:
        start = None
        last = None
        found_segment = False

        for step in range(current, n):
            cur = str(state_series.iloc[step])
            prev = str(state_series.iloc[step - 1])

            # B -> C 作为一段 C 的开始
            if len(cur) > 0 and cur[0] == 'C' and len(prev) > 0 and prev[0] == 'B':
                start = step

            # 相邻相同，输出一下
            elif len(cur) > 0 and len(prev) > 0 and cur[0] == prev[0]:
                #print('problem', step)
                continue

            # C -> A 作为结束
            elif len(cur) > 0 and cur[0] == 'A' and len(prev) > 0 and prev[0] == 'C':
                last = step  # 右开
                found_segment = True
                break

            # C -> over 作为结束
            elif cur == 'over' and len(prev) > 0 and prev[0] == 'C':
                last = step  # 右开
                found_segment = True
                break

            else:
                print(cur, ' ', prev)
                print(current)

        if not found_segment:
            break

        # 有效 C 区间是 [start, last-1]
        segments.append((start, last - 1))
        current = last + 1

    return segments


# ====== 计算单个被试的 saccade 指标，并写出到该被试的文件夹 ======
def process_one_subject(base_dir, subj_id):
    """
    base_dir: G:\\myexperience
    subj_id: 1,2,...,30 这样的整数
    """
    subj_str = str(subj_id)  # 文件夹名和原始文件名都用这个
    subj_dir = os.path.join(base_dir, subj_str)

    fixation_file = os.path.join(subj_dir, "fixation_angle_0.8_time_100ms.xlsx")
    gaze_file = os.path.join(subj_dir, f"{subj_str}.xlsx")

    if not os.path.exists(fixation_file):
        print(f"[被试 {subj_id}] 缺少 fixation 文件：{fixation_file}，跳过。")
        return
    if not os.path.exists(gaze_file):
        print(f"[被试 {subj_id}] 缺少 gaze 文件：{gaze_file}，跳过。")
        return

    print(f"\n=== 开始处理被试 {subj_id} ===")
    print("fixation 文件：", fixation_file)
    print("gaze 文件：", gaze_file)

    # 2. 读取数据
    fix_df = pd.read_excel(fixation_file)
    gaze_df = pd.read_excel(gaze_file)

    if len(fix_df) != len(gaze_df):
        raise ValueError(f"[被试 {subj_id}] fixation 文件与原始 gaze 文件行数不一致！")

    # 3. 如果有 state 列，就按 state 划分 C 段；没有就当成一整段
    if 'state' in fix_df.columns:
        c_segments = split_c_segments(fix_df['state'])
    else:
        print(f"[被试 {subj_id}] 大错误190：没有 state 列，整段视为一个 segment")
        c_segments = [(0, len(fix_df) - 1)]

    print(f"[被试 {subj_id}] 检测到 {len(c_segments)} 个 C 段")

    # 4. 为每个 C 段计算 saccade 指标
    names = []
    saccade_duration_simple = []
    saccade_duration_average = []
    saccade_velocity_simple = []
    saccade_velocity_average = []
    saccade_number = []

    for idx_seg, (seg_start, seg_end) in enumerate(c_segments, start=1):
        names.append(f"C{idx_seg}")

        # 4.1 提取当前 C 段内的 fixation 事件
        fixation_events = extract_fixation_events(
            fix_df['fixation'], seg_start, seg_end
        )

        # 4.2 在这些 fixation 之间计算 saccade
        seg_sacc_durations = []
        seg_sacc_velocities = []

        if len(fixation_events) >= 2:
            for k in range(len(fixation_events) - 1):
                s1, e1 = fixation_events[k]
                s2, e2 = fixation_events[k + 1]

                # 时间：用前一个 fixation 结束时间 -> 下一个 fixation 开始时间
                t1 = fix_df['time'].iloc[e1]
                t2 = fix_df['time'].iloc[s2]
                dur = t2 - t1  # ms
                if dur <= 0:
                    # 时间异常，跳过
                    continue

                # 求两个 fixation 的平均眼球方向
                v1 = mean_eye_dir(gaze_df, s1, e1)
                v2 = mean_eye_dir(gaze_df, s2, e2)
                if v1 is None or v2 is None:
                    continue

                # 夹角 = arccos(v1·v2) (rad)
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)  # 弧度

                # saccade 速度 = 角度 / 时间 （度 / s）
                angle_deg = angle * 180.0 / np.pi  # rad -> deg
                dur_s = dur / 1000.0               # ms -> s
                if dur_s <= 0:
                    continue
                vel = angle_deg / dur_s            # deg/s

                seg_sacc_durations.append(dur)
                seg_sacc_velocities.append(vel)

        # 4.3 汇总当前 C 段的指标
        if len(seg_sacc_durations) == 0:
            saccade_duration_simple.append([])
            saccade_duration_average.append(0.0)
            saccade_velocity_simple.append([])
            saccade_velocity_average.append(0.0)
            saccade_number.append(0)
        else:
            saccade_duration_simple.append(seg_sacc_durations)
            saccade_duration_average.append(float(np.mean(seg_sacc_durations)))
            saccade_velocity_simple.append(seg_sacc_velocities)
            saccade_velocity_average.append(float(np.mean(seg_sacc_velocities)))
            saccade_number.append(len(seg_sacc_durations))

    # 5. 写出结果到该被试的文件夹
    out_df = pd.DataFrame({
        'name': names,
        'saccade_duration_average': saccade_duration_average,
        'saccade_number': saccade_number,
        'saccade_duration_simple': saccade_duration_simple,
        'saccade_velocity_average': saccade_velocity_average,
        'saccade_velocity_simple': saccade_velocity_simple
    })

    out_file = os.path.join(subj_dir, "saccade_calculate_angle_0.8_time_100ms.xlsx")
    out_df.to_excel(out_file, index=False)
    print(f"[被试 {subj_id}] 完成！已保存到：{out_file}")


def main():
    # 注意：这里是所有被试的总目录，不再写到 13
    base_dir = r"G:\myexperience"

    # 一次性跑 1~30 号
    for subj_id in range(1, 30 + 1):
        try:
            process_one_subject(base_dir, subj_id)
        except Exception as e:
            print(f"[被试 {subj_id}] 发生错误：{e}")


if __name__ == "__main__":
    main()
