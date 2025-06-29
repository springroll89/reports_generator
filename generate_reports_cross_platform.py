#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import sys

# ————————————
# 一、配置路径
# ————————————

# 确保使用相对路径，适用于Windows和Mac环境
def get_base_dir():
    """获取基础目录，兼容PyInstaller打包后的路径"""
    if getattr(sys, 'frozen', False):
        # 打包后的程序运行路径
        return os.path.dirname(sys.executable)
    else:
        # 开发环境下运行路径
        return os.path.dirname(os.path.abspath(__file__))

# 设置基础路径
BASE_DIR = get_base_dir()
INPUT_DIR = os.path.join(BASE_DIR, 'reports_input')
OUT_DIR   = os.path.join(BASE_DIR, 'reports_output')

# ————————————
# 二、可选：加载本地仿宋字体
# ————————————
font_file = os.path.join(INPUT_DIR, '.仿宋_GB2312_0.TTF')
if os.path.exists(font_file):
    prop = FontProperties(fname=font_file)
    plt.rcParams['font.family'] = prop.get_name()

# ————————————
# 三、数据清洗函数
# ————————————
def clean_flow_data(data, window_length=11, polyorder=3):
    """
    使用Savitzky-Golay滤波器清洗流量数据
    - window_length: 滤波窗口长度（必须是奇数）
    - polyorder: 多项式阶数
    """
    # 找到第一个非零值的位置
    first_nonzero = np.argmax(data > 0)
    if first_nonzero == 0 and data[0] == 0:
        # 如果所有值都是0，返回原始数据
        first_nonzero = len(data)
    
    # 创建清洗后的数据副本
    cleaned_data = data.copy()
    
    # 只清洗非零部分的数据
    if first_nonzero < len(data):
        nonzero_data = data[first_nonzero:]
        
        # 确保窗口长度是奇数
        if window_length % 2 == 0:
            window_length += 1
        
        # 如果非零数据长度小于窗口长度，调整窗口长度
        if len(nonzero_data) < window_length:
            window_length = len(nonzero_data) if len(nonzero_data) % 2 == 1 else len(nonzero_data) - 1
            if window_length < 5:  # 最小窗口长度
                return cleaned_data  # 数据太少，返回原始数据
        
        # 应用Savitzky-Golay滤波器
        try:
            cleaned_nonzero = savgol_filter(nonzero_data, window_length, polyorder)
            # 确保清洗后的数据不为负
            cleaned_nonzero = np.maximum(cleaned_nonzero, 0)
            # 替换非零部分
            cleaned_data[first_nonzero:] = cleaned_nonzero
        except:
            pass  # 如果滤波失败，保持原始数据
    
    return cleaned_data

# ————————————
# 四、读取温度数据函数
# ————————————
def read_temperature_data(input_dir):
    """
    读取输入目录中的温度CSV文件
    """
    # 查找所有CSV文件（排除基准曲线文件）
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    csv_files = [f for f in csv_files if '基准曲线' not in os.path.basename(f)]
    
    if not csv_files:
        print("⚠️ 未找到温度数据CSV文件")
        return None, None
    
    # 读取第一个找到的CSV文件
    temp_file = csv_files[0]
    print(f"✅ 读取温度数据: {os.path.basename(temp_file)}")
    
    # 尝试不同的编码方式读取
    encodings = ['gbk', 'gb2312', 'utf-8', 'cp1252', 'latin1']
    for encoding in encodings:
        try:
            temp_df = pd.read_csv(temp_file, encoding=encoding)
            # 检查是否包含CH1-CH3列
            required_cols = ['CH1', 'CH2', 'CH3']
            if all(col in temp_df.columns for col in required_cols):
                # 提取温度数据
                temp_data = temp_df[required_cols].values
                # 清洗温度数据（使用较小的窗口）
                smoothed_temp = np.zeros_like(temp_data)
                for i in range(3):
                    smoothed_temp[:, i] = clean_flow_data(temp_data[:, i], window_length=7, polyorder=2)
                
                # 温度数据是2秒采样，返回实际时长
                temp_length = len(smoothed_temp) * 2  # 2秒采样间隔
                print(f"✅ 温度数据点数: {len(smoothed_temp)}, 时长: {temp_length}秒")
                
                return smoothed_temp, temp_length
            else:
                print(f"⚠️ CSV文件缺少必要的列: {required_cols}")
                return None, None
        except:
            continue
    
    print("❌ 无法读取温度数据文件")
    return None, None

# ————————————
# 五、读取基准曲线
# ————————————
baseline_csv = os.path.join(INPUT_DIR, '.基准曲线 1.csv')
base_df = pd.read_csv(baseline_csv)
base_t, base_f = base_df['time'].values, base_df['flow'].values

# ————————————
# 六、读取刻度表
# ————————————
scale_file = os.path.join(INPUT_DIR, '.副本4P、2P22M氧烛分层及刻度.xlsx')
try:
    scale_4p22m = pd.read_excel(scale_file, sheet_name='4P22M')
    scale_2p22m = pd.read_excel(scale_file, sheet_name='2P22M')
    print(f"✅ 成功读取刻度表")
except Exception as e:
    print(f"⚠️ 读取刻度表失败: {e}")
    scale_4p22m = pd.DataFrame()
    scale_2p22m = pd.DataFrame()

# ————————————
# 七、报告通用参数
# ————————————
KEY_TIMES = [60, 100, 142, 360, 1200]   # 关键时间点（秒）

# ————————————
# 八、准备输出目录
# ————————————
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# ————————————
# 九、读取温度数据
# ————————————
temperature_data, temp_length = read_temperature_data(INPUT_DIR)

# ————————————
# 十、扫描并处理每个原始表
# ————————————
pattern = os.path.join(INPUT_DIR, '副本氧气流量采集分析表*.xlsx')
files = glob.glob(pattern)
if not files:
    print(f"❌ 在 {INPUT_DIR} 未找到任何符合 '副本氧气流量采集分析表*.xlsx' 的文件")
    exit(1)

# 处理每个流量采集分析表
for path in files:
    label = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{'='*50}")
    print(f"处理文件: {label}")
    print(f"{'='*50}")
    
    # 1) 读取原始Excel
    raw_df = pd.read_excel(path)

    # 2) 解析采集时间，处理0.5秒采样的问题
    raw_df['采集时间'] = pd.to_datetime(
        raw_df['采集时间'], format='%Y年%m月%d日%H:%M:%S'
    )
    
    # 处理相同时间戳的问题（0.5秒采样）
    time_counts = {}
    adjusted_times = []
    
    for time in raw_df['采集时间']:
        if time in time_counts:
            time_counts[time] += 1
            # 添加0.5秒的偏移
            adjusted_time = time + pd.Timedelta(seconds=0.5 * (time_counts[time] - 1))
            adjusted_times.append(adjusted_time)
        else:
            time_counts[time] = 1
            adjusted_times.append(time)
    
    raw_df['采集时间_调整'] = adjusted_times
    t0 = raw_df['采集时间_调整'].iloc[0]
    raw_df['时间(s)'] = (raw_df['采集时间_调整'] - t0).dt.total_seconds()
    
    # 3) 定义传感器列
    SENSORS = [f"{i}号瞬时流量L/Min" for i in range(1, 5)]
    df = raw_df.copy()
    
    # 4) 数据清洗 - 对每个传感器的流量数据进行清洗（保留前面的0值）
    for sensor in SENSORS:
        original_data = df[sensor].values
        cleaned_data = clean_flow_data(original_data)
        df[sensor] = cleaned_data
    
    # 5) 判断模式（基于清洗后的数据）
    sensor_3_has_data = (df['3号瞬时流量L/Min'] > 0.1).any()
    sensor_4_has_data = (df['4号瞬时流量L/Min'] > 0.1).any()
    
    if sensor_3_has_data and sensor_4_has_data:
        # 4P模式：使用所有4个传感器
        active_sensors = SENSORS
        mode = '4P22M'
        scale_df = scale_4p22m
    else:
        # 2P模式：只使用1号和2号传感器
        active_sensors = SENSORS[:2]
        mode = '2P22M'
        scale_df = scale_2p22m
    
    print(f"✅ 检测到模式: {mode}")
    print(f"✅ 活跃传感器: {active_sensors}")
    
    # 6) 计算平均流量（基于清洗后的数据）
    df['平均流量L/Min'] = df[active_sensors].mean(axis=1)
    df['基准流量L/Min'] = np.interp(df['时间(s)'], base_t, base_f)
    
    # 7) 对齐温度数据
    if temperature_data is not None:
        flow_length = len(df)
        flow_time_length = df['时间(s)'].iloc[-1]  # 流量数据的总时长（秒）
        
        # 初始化温度列为NaN
        df['CH1温度'] = np.nan
        df['CH2温度'] = np.nan
        df['CH3温度'] = np.nan
        
        # 温度数据是2秒采样，需要插值到0.5秒采样的流量数据时间点
        temp_data_points = len(temperature_data)
        temp_time_points = np.arange(0, temp_data_points * 2, 2)  # 0, 2, 4, 6...秒
        
        print(f"温度数据插值: {temp_data_points}个点，时间范围0-{temp_time_points[-1]}秒")
        
        # 对每个温度通道进行插值
        for i, ch in enumerate(['CH1温度', 'CH2温度', 'CH3温度']):
            # 使用线性插值将2秒采样的温度数据映射到0.5秒采样的时间点
            interpolated_temp = np.interp(
                df['时间(s)'].values,  # 目标时间点（0.5秒间隔）
                temp_time_points,       # 原始时间点（2秒间隔）
                temperature_data[:, i], # 原始温度值
                left=np.nan,           # 超出范围的左侧值
                right=np.nan           # 超出范围的右侧值
            )
            df[ch] = interpolated_temp
        
        print(f"✅ 温度数据对齐完成")
    
    # 8) 性能指标分析
    performance_data = []
    for sensor in active_sensors:
        # 启动时长：从开始采集到流量大于零的第一个时间点
        start_idx = df[df[sensor] > 0].index
        start_time = df.loc[start_idx[0], '时间(s)'] if len(start_idx) > 0 else np.nan
        
        # 达峰时长：流量首次达到3.75 L/min的时间
        peak_idx = df[df[sensor] >= 3.75].index
        peak_time = df.loc[peak_idx[0], '时间(s)'] if len(peak_idx) > 0 else np.nan
        
        # 累计流量：对时间和流量的积分
        total_flow = round(trapezoid(df[sensor], df['时间(s)'])/60, 2)
        
        # 达标率：流量大于等于基准曲线的部分占总时间的比例
        compliance_count = (df[sensor] >= df['基准流量L/Min']).sum()
        total_count = len(df)
        compliance_rate = round(compliance_count / total_count * 100, 2) if total_count > 0 else 0
        
        # 产氧时间：从启动时长到流量最后大于零的时刻
        if not np.isnan(start_time):
            last_nonzero_idx = df[df[sensor] > 0].index[-1] if len(df[df[sensor] > 0]) > 0 else 0
            last_time = df.loc[last_nonzero_idx, '时间(s)']
            oxygen_time = round((last_time - start_time) / 60, 2)
        else:
            oxygen_time = 0
        
        performance_data.append({
            '设备': sensor.replace('瞬时流量L/Min', ''),
            '启动时长(秒)': round(start_time, 0) if not np.isnan(start_time) else 0,
            '达峰时长(秒)': round(peak_time, 0) if not np.isnan(peak_time) else 0,
            '累计流量(升)': total_flow,
            '达标率(%)': compliance_rate,
            '产氧时间(分钟)': oxygen_time
        })
    
    # 添加平均值
    avg_start = np.mean([p['启动时长(秒)'] for p in performance_data if p['启动时长(秒)'] > 0])
    avg_peak = np.mean([p['达峰时长(秒)'] for p in performance_data if p['达峰时长(秒)'] > 0])
    avg_total = round(trapezoid(df['平均流量L/Min'], df['时间(s)'])/60, 2)
    
    # 平均达标率
    avg_compliance_count = (df['平均流量L/Min'] >= df['基准流量L/Min']).sum()
    avg_compliance = round(avg_compliance_count / total_count * 100, 2) if total_count > 0 else 0
    
    # 平均产氧时间
    avg_start_idx = df[df['平均流量L/Min'] > 0].index
    if len(avg_start_idx) > 0:
        avg_start_time = df.loc[avg_start_idx[0], '时间(s)']
        avg_last_idx = df[df['平均流量L/Min'] > 0].index[-1]
        avg_last_time = df.loc[avg_last_idx, '时间(s)']
        avg_oxygen_time = round((avg_last_time - avg_start_time) / 60, 2)
    else:
        avg_oxygen_time = 0
    
    performance_data.append({
        '设备': '平均值',
        '启动时长(秒)': round(avg_start, 0) if not np.isnan(avg_start) else 0,
        '达峰时长(秒)': round(avg_peak, 0) if not np.isnan(avg_peak) else 0,
        '累计流量(升)': avg_total,
        '达标率(%)': avg_compliance,
        '产氧时间(分钟)': avg_oxygen_time
    })
    
    performance_df = pd.DataFrame(performance_data)
    
    # 9) 创建Excel写入器
    output_file = os.path.join(OUT_DIR, f"{label}_分析报告.xlsx")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # 9.1) 原始数据Sheet - 只包含流量数据，时间精确到0.5秒
        raw_cols = ['时间(s)']
        raw_cols.extend(SENSORS)
        raw_cols.extend(['平均流量L/Min', '基准流量L/Min'])
        
        raw_data = df[raw_cols].copy()
        # 确保时间列显示0.5秒精度
        raw_data['时间(s)'] = raw_data['时间(s)'].round(1)
        raw_data.to_excel(writer, sheet_name='原始数据', index=False)
        
        # 9.2) 关键点分析
        crit = []
        for t in KEY_TIMES:
            sub = df[df['时间(s)'] <= t]
            if len(sub) == 0:
                continue
            
            # 找到最接近目标时间点的数据
            closest_idx = (df['时间(s)'] - t).abs().idxmin()
            
            rec = {'时间点(s)': t}
            
            # 记录每个传感器在该时间点的瞬时流量
            for s in active_sensors:
                sensor_name = s.replace('瞬时流量L/Min', '')
                rec[f'{sensor_name}瞬时流量(L/Min)'] = round(df.loc[closest_idx, s], 2)
            
            # 记录平均瞬时流量
            rec['平均瞬时流量(L/Min)'] = round(df.loc[closest_idx, '平均流量L/Min'], 2)
            
            # 平均总流量
            rec['平均总流量(L)'] = round(trapezoid(sub['平均流量L/Min'], sub['时间(s)'])/60, 2)
            
            # 基准总流量
            xi = np.linspace(0, t, int(t*2)+1)
            rec['基准总流量(L)'] = round(trapezoid(np.interp(xi, base_t, base_f), xi)/60, 2)
            
            crit.append(rec)
        
        if crit:
            crit_df = pd.DataFrame(crit)
            crit_df.to_excel(writer, sheet_name='关键点分析', index=False)
        
        # 9.3) 性能指标分析
        performance_df.to_excel(writer, sheet_name='性能指标分析', index=False)
        
        # 9.4) 平均异常分析表（0.9阈值）
        # 确定反应的真正结束时间：所有活跃传感器流量都不为0的最后时刻
        actual_end_time = None
        for idx in range(len(df)-1, -1, -1):
            row = df.iloc[idx]
            # 检查所有活跃传感器是否都有流量
            all_sensors_active = all(row[sensor] > 0 for sensor in active_sensors)
            if all_sensors_active:
                actual_end_time = row['时间(s)']
                break
        
        if actual_end_time is None:
            print("⚠️ 未找到有效的反应结束时间")
            actual_end_time = df['时间(s)'].iloc[-1]  # 使用最后时间作为备选
        
        # 窗口时间：从反应开始7分钟后到反应结束前2分钟
        window_start = 7 * 60  # 7分钟 = 420秒
        window_end = actual_end_time - 2 * 60  # 实际结束前2分钟
        
        print(f"\n=== 0.9阈值异常分析 ===")
        print(f"数据采集时间: 0秒 到 {df['时间(s)'].iloc[-1]:.1f}秒")
        print(f"反应实际结束时间: {actual_end_time:.1f}秒 ({actual_end_time/60:.2f}分钟)")
        print(f"窗口范围: {window_start:.1f}秒 到 {window_end:.1f}秒")
        
        if window_end > window_start:
            # 创建一个新的DataFrame只包含窗口内的数据
            window_mask = (df['时间(s)'] >= window_start) & (df['时间(s)'] <= window_end)
            window_df = df[window_mask].copy()
            
            if len(window_df) > 0:
                window_df = window_df.reset_index(drop=True)
                print(f"窗口内数据点数: {len(window_df)}")
                
                # 判断平均流量低于0.9 L/min的时段
                window_df['低于阈值'] = window_df['平均流量L/Min'] < 0.9
                
                # 找出所有异常时间段
                avg_vio_records = []
                in_violation = False
                start_idx = None
                
                for i in range(len(window_df)):
                    current_time = window_df.iloc[i]['时间(s)']
                    current_below = window_df.iloc[i]['低于阈值']
                    
                    if current_below and not in_violation:
                        in_violation = True
                        start_idx = i
                    elif not current_below and in_violation:
                        in_violation = False
                        if start_idx is not None:
                            # 记录异常段
                            start_time = window_df.iloc[start_idx]['时间(s)']
                            end_time = window_df.iloc[i-1]['时间(s)']
                            
                            # 计算累积流量
                            start_cumul = round(trapezoid(
                                df[df['时间(s)'] <= start_time]['平均流量L/Min'],
                                df[df['时间(s)'] <= start_time]['时间(s)']
                            )/60, 2)
                            
                            end_cumul = round(trapezoid(
                                df[df['时间(s)'] <= end_time]['平均流量L/Min'],
                                df[df['时间(s)'] <= end_time]['时间(s)']
                            )/60, 2)
                            
                            avg_vio_records.append({
                                '开始时间(秒)': round(start_time, 0),
                                '结束时间(秒)': round(end_time, 0),
                                '持续时间(秒)': round(end_time - start_time, 0),
                                '起点累积流量(升)': start_cumul,
                                '终点累积流量(升)': end_cumul
                            })
                            print(f"记录异常: {start_time:.1f}s - {end_time:.1f}s")
                        start_idx = None
                
                # 处理最后的异常段（如果在窗口内结束）
                if in_violation and start_idx is not None:
                    start_time = window_df.iloc[start_idx]['时间(s)']
                    end_time = window_df.iloc[-1]['时间(s)']
                    
                    # 只有当异常完全在窗口内时才记录
                    if end_time <= window_end:
                        start_cumul = round(trapezoid(
                            df[df['时间(s)'] <= start_time]['平均流量L/Min'],
                            df[df['时间(s)'] <= start_time]['时间(s)']
                        )/60, 2)
                        
                        end_cumul = round(trapezoid(
                            df[df['时间(s)'] <= end_time]['平均流量L/Min'],
                            df[df['时间(s)'] <= end_time]['时间(s)']
                        )/60, 2)
                        
                        avg_vio_records.append({
                            '开始时间(秒)': round(start_time, 0),
                            '结束时间(秒)': round(end_time, 0),
                            '持续时间(秒)': round(end_time - start_time, 0),
                            '起点累积流量(升)': start_cumul,
                            '终点累积流量(升)': end_cumul
                        })
                        print(f"记录最后异常: {start_time:.1f}s - {end_time:.1f}s")
                
                if avg_vio_records:
                    print(f"\n✅ 共找到 {len(avg_vio_records)} 个窗口内的异常时段")
                    avg_vio_df = pd.DataFrame(avg_vio_records)
                    avg_vio_df.to_excel(writer, sheet_name='平均异常分析表_0.9阈值', index=False)
                else:
                    print("\n窗口内未发现异常")
            else:
                print("窗口内无数据")
        else:
            print("窗口时间无效（反应时间太短）")
        
        # 9.5) 各传感器异常分析
        for sensor in active_sensors:
            sensor_name = sensor.replace('瞬时流量L/Min', '')
            
            # 判断低于基准的时间段
            df['低于基准'] = df[sensor] < df['基准流量L/Min']
            
            # 找出所有不达标时间段
            violations = []
            in_violation = False
            start_idx = None
            
            for idx in range(len(df)):
                if df.iloc[idx]['低于基准'] and not in_violation:
                    in_violation = True
                    start_idx = idx
                elif not df.iloc[idx]['低于基准'] and in_violation:
                    in_violation = False
                    if start_idx is not None:
                        violations.append((start_idx, idx - 1))
                    start_idx = None
            
            # 如果最后还在违规中
            if in_violation and start_idx is not None:
                violations.append((start_idx, len(df) - 1))
            
            # 分析每个不达标时间段
            vio_records = []
            for start, end in violations:
                sub = df.iloc[start:end+1]
                if len(sub) < 2:
                    continue
                
                # 计算流量差异（异常期间低于基准的累积差值）
                diff = sub['基准流量L/Min'] - sub[sensor]
                flow_diff = round(trapezoid(np.maximum(diff, 0), sub['时间(s)'])/60, 2)
                
                # 累积流量
                start_cumul = round(trapezoid(
                    df[df['时间(s)'] <= sub['时间(s)'].iloc[0]][sensor],
                    df[df['时间(s)'] <= sub['时间(s)'].iloc[0]]['时间(s)']
                )/60, 2)
                
                end_cumul = round(trapezoid(
                    df[df['时间(s)'] <= sub['时间(s)'].iloc[-1]][sensor],
                    df[df['时间(s)'] <= sub['时间(s)'].iloc[-1]]['时间(s)']
                )/60, 2)
                
                vio_records.append({
                    '开始时间(秒)': round(sub['时间(s)'].iloc[0], 0),
                    '结束时间(秒)': round(sub['时间(s)'].iloc[-1], 0),
                    '持续时间(秒)': round(sub['时间(s)'].iloc[-1] - sub['时间(s)'].iloc[0], 0),
                    '流量差异(升)': flow_diff,
                    '起点累积流量(升)': start_cumul,
                    '终点累积流量(升)': end_cumul
                })
            
            if vio_records:
                vio_df = pd.DataFrame(vio_records)
                sheet_name = f'异常分析_{sensor_name}'
                vio_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 9.6) 异常分析_平均值
        # 使用平均流量进行异常分析
        df['低于基准'] = df['平均流量L/Min'] < df['基准流量L/Min']
        
        # 找出所有不达标时间段
        violations = []
        in_violation = False
        start_idx = None
        
        for idx in range(len(df)):
            if df.iloc[idx]['低于基准'] and not in_violation:
                in_violation = True
                start_idx = idx
            elif not df.iloc[idx]['低于基准'] and in_violation:
                in_violation = False
                if start_idx is not None:
                    violations.append((start_idx, idx - 1))
                start_idx = None
        
        # 如果最后还在违规中
        if in_violation and start_idx is not None:
            violations.append((start_idx, len(df) - 1))
        
        # 分析每个不达标时间段
        vio_records = []
        for start, end in violations:
            sub = df.iloc[start:end+1]
            if len(sub) < 2:
                continue
            
            # 计算流量差异
            diff = sub['基准流量L/Min'] - sub['平均流量L/Min']
            flow_diff = round(trapezoid(np.maximum(diff, 0), sub['时间(s)'])/60, 2)
            
            # 累积流量
            start_cumul = round(trapezoid(
                df[df['时间(s)'] <= sub['时间(s)'].iloc[0]]['平均流量L/Min'],
                df[df['时间(s)'] <= sub['时间(s)'].iloc[0]]['时间(s)']
            )/60, 2)
            
            end_cumul = round(trapezoid(
                df[df['时间(s)'] <= sub['时间(s)'].iloc[-1]]['平均流量L/Min'],
                df[df['时间(s)'] <= sub['时间(s)'].iloc[-1]]['时间(s)']
            )/60, 2)
            
            vio_records.append({
                '开始时间(秒)': round(sub['时间(s)'].iloc[0], 0),
                '结束时间(秒)': round(sub['时间(s)'].iloc[-1], 0),
                '持续时间(秒)': round(sub['时间(s)'].iloc[-1] - sub['时间(s)'].iloc[0], 0),
                '流量差异(升)': flow_diff,
                '起点累积流量(升)': start_cumul,
                '终点累积流量(升)': end_cumul
            })
        
        if vio_records:
            vio_df = pd.DataFrame(vio_records)
            vio_df.to_excel(writer, sheet_name='异常分析_平均值', index=False)
        
        # 9.7) 燃烧定位表
        if not scale_df.empty:
            # 首先计算总产氧量
            total_oxygen = round(trapezoid(df['平均流量L/Min'], df['时间(s)'])/60, 2)
            
            # 燃烧定位数据
            burn_data = []
            for idx, row in df.iterrows():
                t = row['时间(s)']
                # 计算累积产氧量
                sub = df[df['时间(s)'] <= t]
                oxygen_produced = round(trapezoid(sub['平均流量L/Min'], sub['时间(s)'])/60, 2)
                
                # 计算累积产氧量占总产氧量的百分比
                if total_oxygen > 0:
                    oxygen_percentage = oxygen_produced / total_oxygen
                else:
                    oxygen_percentage = 0
                
                # 根据百分比在刻度表中插值计算燃烧深度
                depth = 0
                position = "未开始"
                
                if '有效燃烧百分比' in scale_df.columns and '刻度值/mm' in scale_df.columns:
                    # 获取刻度表数据（排除表头）
                    scale_data = scale_df[scale_df['有效燃烧百分比'].notna()].copy()
                    
                    if len(scale_data) > 0 and oxygen_percentage > 0:
                        # 使用插值计算燃烧深度
                        percentages = scale_data['有效燃烧百分比'].values
                        depths = scale_data['刻度值/mm'].values
                        
                        # 如果百分比超出范围，使用边界值
                        if oxygen_percentage <= percentages.min():
                            depth = depths[0]
                        elif oxygen_percentage >= percentages.max():
                            depth = depths[-1]
                        else:
                            # 线性插值
                            depth = np.interp(oxygen_percentage, percentages, depths)
                        
                        depth = round(depth, 1)
                        
                        # 根据深度确定位置
                        # 查找最接近的刻度值对应的位置
                        closest_idx = np.abs(scale_data['刻度值/mm'] - depth).idxmin()
                        if '大致位置' in scale_data.columns:
                            position = scale_data.loc[closest_idx, '大致位置']
                
                elif oxygen_produced > 0:
                    # 如果没有有效燃烧百分比列，使用简单的线性估算
                    # 假设总深度200mm
                    depth = round(oxygen_percentage * 200, 1)
                    if depth < 50:
                        position = "A、B层"
                    elif depth < 100:
                        position = "C1层"
                    elif depth < 150:
                        position = "C2层"
                    else:
                        position = "C3层"
                
                burn_data.append({
                    '时间(s)': round(t, 0),
                    '产氧量(L)': oxygen_produced,
                    '燃烧深度(mm)': depth,
                    '大致位置': position
                })
            
            # 创建燃烧定位表（关键时间点）
            key_times = [0, 60, 120, 180, 240, 300, 360, 600, 900, 1200]
            key_burn_data = []
            
            for t in key_times:
                # 找到最接近的时间点
                if t == 0:
                    key_burn_data.append(burn_data[0])
                else:
                    # 考虑0.5秒采样，找到最接近的索引
                    target_idx = int(t * 2)  # t秒对应的索引（0.5秒采样）
                    if target_idx < len(burn_data):
                        key_burn_data.append(burn_data[target_idx])
                    elif len(burn_data) > 0:
                        # 如果超出范围，使用最后一个数据
                        key_burn_data.append(burn_data[-1])
            
            burn_df = pd.DataFrame(key_burn_data)
            burn_df.to_excel(writer, sheet_name='燃烧定位表', index=False)
            
            # 完整的燃烧定位数据
            full_burn_df = pd.DataFrame(burn_data)
            full_burn_df.to_excel(writer, sheet_name='燃烧定位数据', index=False)
        
        # 9.8) 创建曲线图Sheet - 显示所有传感器流量曲线
        # 为图表数据创建每秒一个点的数据（用于图表显示）
        # 只选择整数秒的数据点
        chart_data = df[df['时间(s)'] % 1 == 0].copy()
        
        # 如果没有整数秒的数据，则取最接近整数秒的数据
        if len(chart_data) == 0:
            # 创建整数秒的时间点
            max_time = int(df['时间(s)'].max())
            time_points = list(range(0, max_time + 1))
            
            chart_rows = []
            for t in time_points:
                # 找到最接近这个整数秒的数据点
                closest_idx = (df['时间(s)'] - t).abs().idxmin()
                row_data = df.loc[closest_idx].copy()
                row_data['时间(s)'] = t  # 设置为整数秒
                chart_rows.append(row_data)
            
            chart_data = pd.DataFrame(chart_rows)
        
        # 创建流量曲线图数据
        chart_cols = ['时间(s)']
        chart_cols.extend(active_sensors)  # 添加活跃的传感器列
        chart_cols.extend(['平均流量L/Min', '基准流量L/Min'])
        
        chart_df = chart_data[chart_cols].copy()
        # 确保时间是整数
        chart_df['时间(s)'] = chart_df['时间(s)'].astype(int)
        chart_df.to_excel(writer, sheet_name='曲线图', index=False)
        
        # 在Excel中创建流量曲线图表
        worksheet = writer.sheets['曲线图']
        
        # 创建流量图表
        chart1 = workbook.add_chart({'type': 'line'})
        
        # 添加各个传感器的流量曲线
        colors = ['blue', 'green', 'orange', 'purple']
        for i, sensor in enumerate(active_sensors):
            sensor_name = sensor.replace('瞬时流量L/Min', '')
            chart1.add_series({
                'name': f'{sensor_name}流量',
                'categories': ['曲线图', 1, 0, len(chart_df), 0],  # 时间列
                'values': ['曲线图', 1, i+1, len(chart_df), i+1],  # 传感器流量列
                'line': {'color': colors[i], 'width': 1.5}
            })
        
        # 添加平均流量系列
        chart1.add_series({
            'name': '平均流量',
            'categories': ['曲线图', 1, 0, len(chart_df), 0],
            'values': ['曲线图', 1, len(active_sensors)+1, len(chart_df), len(active_sensors)+1],
            'line': {'color': 'black', 'width': 2.5}
        })
        
        # 添加基准流量系列
        chart1.add_series({
            'name': '基准流量',
            'categories': ['曲线图', 1, 0, len(chart_df), 0],
            'values': ['曲线图', 1, len(active_sensors)+2, len(chart_df), len(active_sensors)+2],
            'line': {'color': 'red', 'width': 2.5, 'dash_type': 'dash'}
        })
        
        # 设置图表属性
        chart1.set_title({'name': '流量曲线对比图'})
        chart1.set_x_axis({'name': '时间 (秒)'})
        chart1.set_y_axis({'name': '流量 (L/Min)'})
        chart1.set_size({'width': 720, 'height': 480})
        worksheet.insert_chart('H2', chart1)
        
        # 9.9) 创建流量-温度综合分析图（如果有温度数据）
        if temperature_data is not None:
            print("创建流量-温度综合分析图...")
            
            # 创建综合分析数据
            temp_chart_cols = ['时间(s)', '平均流量L/Min', 'CH1温度', 'CH2温度', 'CH3温度']
            
            # 检查是否所有列都存在
            missing_cols = [col for col in temp_chart_cols if col not in chart_data.columns]
            if missing_cols:
                print(f"警告：缺少列 {missing_cols}")
                # 只使用存在的列
                temp_chart_cols = [col for col in temp_chart_cols if col in chart_data.columns]
            
            if len(temp_chart_cols) >= 2:  # 至少需要时间和一个数据列
                temp_chart_df = chart_data[temp_chart_cols].copy()
                temp_chart_df['时间(s)'] = temp_chart_df['时间(s)'].astype(int)
                temp_chart_df.to_excel(writer, sheet_name='流量-温度综合分析', index=False)
                
                # 创建综合图表
                worksheet_temp = writer.sheets['流量-温度综合分析']
                chart3 = workbook.add_chart({'type': 'line'})
                
                # 添加平均流量系列（左侧Y轴）
                if '平均流量L/Min' in temp_chart_df.columns:
                    chart3.add_series({
                        'name': '平均流量',
                        'categories': ['流量-温度综合分析', 1, 0, len(temp_chart_df), 0],
                        'values': ['流量-温度综合分析', 1, 1, len(temp_chart_df), 1],
                        'line': {'color': 'blue', 'width': 2.5}
                    })
                
                # 添加温度系列（右侧Y轴）
                temp_colors = ['red', 'green', 'orange']
                temp_names = ['CH1温度', 'CH2温度', 'CH3温度']
                col_idx = 2  # 从第3列开始（索引2）
                
                for name, color in zip(temp_names, temp_colors):
                    if name in temp_chart_df.columns:
                        chart3.add_series({
                            'name': name,
                            'categories': ['流量-温度综合分析', 1, 0, len(temp_chart_df), 0],
                            'values': ['流量-温度综合分析', 1, col_idx, len(temp_chart_df), col_idx],
                            'line': {'color': color, 'width': 2},
                            'y2_axis': True  # 右侧Y轴
                        })
                        col_idx += 1
                
                # 设置图表属性
                chart3.set_title({'name': '流量-温度综合分析图'})
                chart3.set_x_axis({'name': '时间 (秒)'})
                chart3.set_y_axis({'name': '流量 (L/Min)', 'major_gridlines': {'visible': True}})
                chart3.set_y2_axis({'name': '温度 (°C)'})
                chart3.set_size({'width': 720, 'height': 480})
                worksheet_temp.insert_chart('G2', chart3)
                
                print("✅ 流量-温度综合分析图创建成功")
            else:
                print("⚠️ 数据不足，无法创建流量-温度综合分析图")
        else:
            print("⚠️ 无温度数据，跳过流量-温度综合分析图")
        
        # 9.10) 产氧-深度曲线
        if 'burn_data' in locals() and len(burn_data) > 0:
            # 创建产氧深度数据（每10秒一个点）
            depth_data = []
            for i in range(0, len(burn_data), 20):  # 每10秒取一个点（0.5秒采样）
                depth_data.append(burn_data[i])
            
            depth_df = pd.DataFrame(depth_data)
            depth_df.to_excel(writer, sheet_name='产氧-深度曲线', index=False)
            
            # 创建深度图表
            worksheet_depth = writer.sheets['产氧-深度曲线']
            chart2 = workbook.add_chart({'type': 'line'})
            
            # 添加产氧量系列（左侧Y轴）
            chart2.add_series({
                'name': '产氧量',
                'categories': ['产氧-深度曲线', 1, 0, len(depth_df), 0],  # 时间列
                'values': ['产氧-深度曲线', 1, 1, len(depth_df), 1],     # 产氧量列
                'line': {'color': 'blue', 'width': 2},
                'y_axis': 1  # 左侧Y轴
            })
            
            # 添加燃烧深度系列（右侧Y轴）
            chart2.add_series({
                'name': '燃烧深度',
                'categories': ['产氧-深度曲线', 1, 0, len(depth_df), 0],  # 时间列
                'values': ['产氧-深度曲线', 1, 2, len(depth_df), 2],     # 深度列
                'line': {'color': 'red', 'width': 2},
                'y2_axis': True  # 右侧Y轴
            })
            
            chart2.set_title({'name': '产氧量-燃烧深度关系图'})
            chart2.set_x_axis({'name': '时间 (秒)'})
            chart2.set_y_axis({'name': '产氧量 (L)', 'major_gridlines': {'visible': True}})
            chart2.set_y2_axis({'name': '燃烧深度 (mm)'})
            chart2.set_size({'width': 720, 'height': 480})
            worksheet_depth.insert_chart('E2', chart2)
    
    print(f"✅ 生成报告: {output_file}")

print(f"\n{'='*50}")
print(f"所有报告生成完成！")
print(f"输出目录: {OUT_DIR}")
print(f"{'='*50}")