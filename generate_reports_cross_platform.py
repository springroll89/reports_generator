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
from scipy import stats
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
# 四、平稳性分析函数
# ————————————
def calculate_cumulative_flow(df, active_sensors):
    """计算累积流量（升）- 基于所有活跃传感器的总流量"""
    # 计算所有活跃传感器的总流量
    total_flow = df[active_sensors].sum(axis=1)
    # 使用梯形积分计算累积流量
    cumulative = np.zeros(len(df))
    for i in range(1, len(df)):
        # 计算到当前时间点的累积流量
        cumulative[i] = trapezoid(total_flow.values[:i+1], df['时间(s)'].values[:i+1]) / 60
    return cumulative

def find_stable_period(df, cumulative_flow, reaction_end_time):
    """确定平稳期边界
    平稳期起点：所有活跃传感器的总产氧量达到20L
    平稳期终点：反应结束前120秒
    """
    # 找到累积流量达到20L的时间点
    start_indices = np.where(cumulative_flow >= 20)[0]
    if len(start_indices) == 0:
        print("⚠️ 总产氧量未达到20L，无法进行平稳性分析")
        return None, None
    
    start_idx = start_indices[0]
    start_time = df.iloc[start_idx]['时间(s)']
    
    # 平稳期终点：反应结束前120秒（使用传入的反应结束时间）
    end_time = reaction_end_time - 120
    
    if end_time <= start_time:
        print("⚠️ 反应时间过短，无法确定有效的平稳期")
        return None, None
    
    return start_time, end_time

def extract_stable_data(df, start_time, end_time):
    """提取平稳期数据"""
    mask = (df['时间(s)'] >= start_time) & (df['时间(s)'] <= end_time)
    return df[mask].copy()

def linear_fit(stable_data):
    """对平稳期数据进行线性拟合"""
    x = stable_data['时间(s)'].values
    y = stable_data['平均流量L/Min'].values
    
    # 使用scipy进行线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

def calculate_deviations(stable_data, fit_params):
    """计算实际值与拟合值的偏差"""
    x = stable_data['时间(s)'].values
    y_actual = stable_data['平均流量L/Min'].values
    y_fitted = fit_params['slope'] * x + fit_params['intercept']
    
    deviations = y_actual - y_fitted
    return deviations, y_fitted

def calculate_stability_metrics(stable_data, deviations, cumulative_flow, df, active_sensors):
    """计算稳定性指标"""
    mean_flow = stable_data['平均流量L/Min'].mean()
    
    # 相对基准线变异系数
    cv = np.std(deviations) / mean_flow if mean_flow > 0 else np.nan
    
    # 最大偏差幅度
    max_deviation = np.max(np.abs(deviations))
    
    # 超调时间占比（使用15%阈值）
    threshold = mean_flow * 0.15
    exceed_count = np.sum(np.abs(deviations) > threshold)
    exceed_ratio = exceed_count / len(deviations) * 100 if len(deviations) > 0 else 0
    
    # 平稳期产氧效率（基于总流量）
    start_idx = stable_data.index[0]
    end_idx = stable_data.index[-1]
    start_cumul = cumulative_flow[start_idx]
    end_cumul = cumulative_flow[end_idx]
    duration = (stable_data['时间(s)'].iloc[-1] - stable_data['时间(s)'].iloc[0]) / 60  # 转换为分钟
    efficiency = (end_cumul - start_cumul) / duration if duration > 0 else 0
    
    # 启动响应时间（达到20L的时间）- 基于总流量
    start_time_20L = stable_data['时间(s)'].iloc[0]
    
    return {
        'cv': cv,
        'max_deviation': max_deviation,
        'exceed_ratio': exceed_ratio,
        'exceed_threshold': threshold,
        'mean_flow': mean_flow,
        'efficiency': efficiency,
        'start_time_20L': start_time_20L,
        'start_cumul': start_cumul,
        'end_cumul': end_cumul
    }

def sliding_window_analysis(stable_data, window_size=60, step_size=10):
    """滑动窗口分析"""
    results = []
    time_array = stable_data['时间(s)'].values
    flow_array = stable_data['平均流量L/Min'].values
    
    # 确保窗口不超过数据长度
    if len(stable_data) * 0.5 < window_size:  # 0.5秒采样
        window_size = int(len(stable_data) * 0.5 / 2)  # 使用一半的数据长度
        step_size = max(5, window_size // 6)
    
    # 将时间窗口转换为索引窗口
    window_indices = int(window_size / 0.5)  # 0.5秒采样
    step_indices = int(step_size / 0.5)
    
    for i in range(0, len(stable_data) - window_indices, step_indices):
        window_data = flow_array[i:i+window_indices]
        window_time = time_array[i:i+window_indices]
        
        # 计算窗口内的CV
        window_cv = np.std(window_data) / np.mean(window_data) if np.mean(window_data) > 0 else 0
        
        # 计算窗口内的最大斜率（使用差分）
        if len(window_data) > 1:
            slopes = np.diff(window_data) / 0.5  # 0.5秒采样间隔
            max_slope = np.max(np.abs(slopes))
        else:
            max_slope = 0
        
        # 稳定性评价
        if window_cv < 0.01:
            stability = "优秀"
        elif window_cv < 0.02:
            stability = "良好"
        elif window_cv < 0.04:
            stability = "一般"
        else:
            stability = "较差"
        
        results.append({
            '时间段': f"{window_time[0]:.0f}-{window_time[-1]:.0f}s",
            '局部CV': round(window_cv, 4),
            '局部最大斜率': round(max_slope, 4),
            '稳定性评价': stability
        })
    
    return results

def evaluate_metric(value, thresholds):
    """根据阈值评价指标"""
    if value < thresholds[0]:
        return "优秀"
    elif value < thresholds[1]:
        return "良好"
    elif value < thresholds[2]:
        return "一般"
    else:
        return "较差"

# ————————————
# 五、读取温度数据函数
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
# 六、读取基准曲线
# ————————————
baseline_csv = os.path.join(INPUT_DIR, '.基准曲线 1.csv')
base_df = pd.read_csv(baseline_csv)
base_t, base_f = base_df['time'].values, base_df['flow'].values

# ————————————
# 七、读取刻度表
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
# 八、报告通用参数
# ————————————
KEY_TIMES = [10, 80, 150, 360, 420, 1200, 1248, 1320]   # 关键时间点（秒）

# ————————————
# 九、准备输出目录
# ————————————
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# ————————————
# 十、读取温度数据
# ————————————
temperature_data, temp_length = read_temperature_data(INPUT_DIR)

# ————————————
# 十一、扫描并处理每个原始表
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
    
    # 6.5) 统一定义反应起止时间
    # 反应开始时间：固定为0秒（数据采集开始）
    REACTION_START_TIME = 0.0
    
    # 反应结束时间：平均流量最后一次大于0的时刻
    nonzero_mask = df['平均流量L/Min'] > 0
    if nonzero_mask.any():
        REACTION_END_TIME = df[nonzero_mask]['时间(s)'].iloc[-1]
    else:
        REACTION_END_TIME = df['时间(s)'].iloc[-1]
    
    # 反应总时长（秒）
    REACTION_DURATION = REACTION_END_TIME - REACTION_START_TIME
    
    print(f"\n✅ 反应时间定义：")
    print(f"   - 开始时间: {REACTION_START_TIME}秒")
    print(f"   - 结束时间: {REACTION_END_TIME:.1f}秒")
    print(f"   - 反应时长: {REACTION_DURATION:.1f}秒 ({REACTION_DURATION/60:.2f}分钟)")
    
    # 6.6) 截断数据 - 去掉反应结束后的无效数据
    # 保存原始数据长度信息
    original_length = len(df)
    original_end_time = df['时间(s)'].iloc[-1]
    
    # 截断数据框，只保留到反应结束时间的数据
    # 找到最接近反应结束时间的索引
    end_index = df[df['时间(s)'] <= REACTION_END_TIME].index[-1]
    df = df.loc[:end_index].copy()
    
    # 打印截断信息
    if len(df) < original_length:
        print(f"✅ 数据截断：原始数据 {original_length} 个点（到 {original_end_time:.1f}秒），截断后 {len(df)} 个点（到 {df['时间(s)'].iloc[-1]:.1f}秒）")
        print(f"   - 删除了 {original_length - len(df)} 个无效数据点")
    
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
    
    # 8) 计算累积流量（用于平稳性分析）- 基于所有活跃传感器的总流量
    # 注意：此时数据已经截断，只包含有效反应时间内的数据
    cumulative_flow = calculate_cumulative_flow(df, active_sensors)
    df['累积总流量L'] = cumulative_flow
    
    # 9) 性能指标分析
    performance_data = []
    for sensor in active_sensors:
        # 启动时长：从反应开始（0秒）到流量大于零的第一个时间点
        start_idx = df[df[sensor] > 0].index
        if len(start_idx) > 0:
            start_time = df.loc[start_idx[0], '时间(s)'] - REACTION_START_TIME
        else:
            start_time = np.nan
        
        # 达峰时长：流量首次达到3.75 L/min的时间
        peak_idx = df[df[sensor] >= 3.75].index
        peak_time = df.loc[peak_idx[0], '时间(s)'] if len(peak_idx) > 0 else np.nan
        
        # 累计流量：对时间和流量的积分（整个反应期间）
        total_flow = round(trapezoid(df[sensor], df['时间(s)'])/60, 2)
        
        # 达标率：流量大于等于基准曲线的部分占总时间的比例
        compliance_count = (df[sensor] >= df['基准流量L/Min']).sum()
        total_count = len(df)
        compliance_rate = round(compliance_count / total_count * 100, 2) if total_count > 0 else 0
        
        # 产氧时间：整个反应时长（使用统一定义）
        oxygen_time = round(REACTION_DURATION / 60, 2)
        
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
    
    # 平均产氧时间（使用统一定义）
    avg_oxygen_time = round(REACTION_DURATION / 60, 2)
    
    performance_data.append({
        '设备': '平均值',
        '启动时长(秒)': round(avg_start, 0) if not np.isnan(avg_start) else 0,
        '达峰时长(秒)': round(avg_peak, 0) if not np.isnan(avg_peak) else 0,
        '累计流量(升)': avg_total,
        '达标率(%)': avg_compliance,
        '产氧时间(分钟)': avg_oxygen_time
    })
    
    performance_df = pd.DataFrame(performance_data)
    
    # 10) 创建Excel写入器
    output_file = os.path.join(OUT_DIR, f"{label}_分析报告.xlsx")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # 10.1) 原始数据Sheet - 只包含流量数据，时间精确到0.5秒
        raw_cols = ['时间(s)']
        raw_cols.extend(SENSORS)
        raw_cols.extend(['平均流量L/Min', '基准流量L/Min', '累积总流量L'])
        
        raw_data = df[raw_cols].copy()
        # 确保时间列显示0.5秒精度
        raw_data['时间(s)'] = raw_data['时间(s)'].round(1)
        raw_data.to_excel(writer, sheet_name='原始数据', index=False)
        
        # 10.2) 关键点分析
        crit = []
        for t in KEY_TIMES:
            # 只分析在反应时间范围内的关键点
            if t > REACTION_END_TIME:
                continue
                
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
        
        # 10.3) 性能指标分析
        performance_df.to_excel(writer, sheet_name='性能指标分析', index=False)
        
        # 10.3.1) 点火测试记录数据
        ignition_test_data = []
        
        # 计算各传感器的点火测试数据
        for i, sensor in enumerate(active_sensors):
            sensor_name = sensor.replace('瞬时流量L/Min', '')
            
            # 从性能指标中获取数据
            perf_data = performance_data[i]  # 对应传感器的性能数据
            
            # 供氧启动时长
            startup_time = perf_data['启动时长(秒)']
            
            # 达峰时长
            peak_time = perf_data['达峰时长(秒)']
            
            # 峰值维持时长：从达峰时间到流量第一次小于3.1 L/min的时长
            peak_duration = 0
            if peak_time > 0:
                # 找到达峰后流量第一次小于3.1的时间
                peak_idx = df[df[sensor] >= 3.75].index[0]
                # 从达峰点开始查找
                after_peak = df.loc[peak_idx:]
                below_3_1 = after_peak[after_peak[sensor] < 3.1]
                if len(below_3_1) > 0:
                    end_peak_time = df.loc[below_3_1.index[0], '时间(s)']
                    peak_duration = round(end_peak_time - peak_time, 0)
                else:
                    # 如果一直没有低于3.1，则到反应结束
                    last_nonzero = df[df[sensor] > 0].index[-1] if len(df[df[sensor] > 0]) > 0 else peak_idx
                    end_peak_time = df.loc[last_nonzero, '时间(s)']
                    peak_duration = round(end_peak_time - peak_time, 0)
            
            # 从启动到峰值结束供氧总量
            peak_end_oxygen = 0
            if peak_time > 0 and peak_duration >= 0:
                peak_end_time = peak_time + peak_duration
                # 计算到峰值结束时的累积流量
                mask = df['时间(s)'] <= peak_end_time
                peak_end_oxygen = round(trapezoid(df[mask][sensor], df[mask]['时间(s)'])/60, 2)
            
            # 反应总时长（使用统一定义）
            total_reaction_time = round(REACTION_DURATION, 0)
            
            # 总累积供氧
            total_oxygen = perf_data['累计流量(升)']
            
            ignition_test_data.append({
                '传感器': sensor_name,
                '供氧启动时长(秒)': startup_time,
                '达峰时长(秒)': peak_time,
                '峰值维持时长(秒)': peak_duration,
                '从启动到峰值结束供氧总量(升)': peak_end_oxygen,
                '反应总时长(秒)': total_reaction_time,
                '总累积供氧(升)': total_oxygen
            })
        
        # 添加平均值数据（从性能指标分析表获取）
        avg_perf_data = performance_data[-1]  # 最后一行是平均值
        
        # 计算平均流量的峰值维持时长
        avg_peak_duration = 0
        avg_peak_time = avg_perf_data['达峰时长(秒)']
        if avg_peak_time > 0:
            # 找到平均流量达峰后第一次小于3.1的时间
            avg_peak_idx = df[df['平均流量L/Min'] >= 3.75].index
            if len(avg_peak_idx) > 0:
                peak_idx = avg_peak_idx[0]
                peak_time_value = df.loc[peak_idx, '时间(s)']
                after_peak = df.loc[peak_idx:]
                below_3_1 = after_peak[after_peak['平均流量L/Min'] < 3.1]
                if len(below_3_1) > 0:
                    end_peak_time = df.loc[below_3_1.index[0], '时间(s)']
                    avg_peak_duration = round(end_peak_time - peak_time_value, 0)
                else:
                    last_nonzero = df[df['平均流量L/Min'] > 0].index[-1]
                    end_peak_time = df.loc[last_nonzero, '时间(s)']
                    avg_peak_duration = round(end_peak_time - peak_time_value, 0)
        
        # 计算平均流量的峰值结束供氧总量
        avg_peak_end_oxygen = 0
        if avg_peak_time > 0 and avg_peak_duration >= 0:
            peak_end_time = avg_peak_time + avg_peak_duration
            # 计算到峰值结束时的累积流量（平均流量）
            mask = df['时间(s)'] <= peak_end_time
            avg_peak_end_oxygen = round(trapezoid(df[mask]['平均流量L/Min'], df[mask]['时间(s)'])/60, 2)
        
        ignition_test_data.append({
            '传感器': '平均值',
            '供氧启动时长(秒)': avg_perf_data['启动时长(秒)'],
            '达峰时长(秒)': avg_perf_data['达峰时长(秒)'],
            '峰值维持时长(秒)': avg_peak_duration,
            '从启动到峰值结束供氧总量(升)': avg_peak_end_oxygen,
            '反应总时长(秒)': round(REACTION_DURATION, 0),
            '总累积供氧(升)': avg_perf_data['累计流量(升)']
        })
        
        # 添加所有活跃传感器的总和数据
        total_startup = round(np.mean([d['供氧启动时长(秒)'] for d in ignition_test_data[:-1] if d['供氧启动时长(秒)'] > 0]), 0)
        total_peak = round(np.mean([d['达峰时长(秒)'] for d in ignition_test_data[:-1] if d['达峰时长(秒)'] > 0]), 0)
        
        # 计算总的峰值维持时长（基于平均流量）
        total_peak_duration = avg_peak_duration  # 使用平均流量的峰值维持时长
        
        # 计算所有传感器的总供氧量（到峰值结束）
        total_peak_end_oxygen = 0
        if total_peak > 0 and total_peak_duration >= 0:
            peak_end_time = total_peak + total_peak_duration
            # 计算所有活跃传感器的总流量
            mask = df['时间(s)'] <= peak_end_time
            for sensor in active_sensors:
                total_peak_end_oxygen += trapezoid(df[mask][sensor], df[mask]['时间(s)'])/60
            total_peak_end_oxygen = round(total_peak_end_oxygen, 2)
        
        # 反应总时长（使用统一定义）
        total_reaction = round(REACTION_DURATION, 0)
        
        # 总累积供氧（所有传感器的总和）
        total_cumulative = round(sum([d['总累积供氧(升)'] for d in ignition_test_data[:-1]]), 2)
        
        ignition_test_data.append({
            '传感器': f'总计({len(active_sensors)}个传感器)',
            '供氧启动时长(秒)': total_startup,
            '达峰时长(秒)': total_peak,
            '峰值维持时长(秒)': total_peak_duration,
            '从启动到峰值结束供氧总量(升)': total_peak_end_oxygen,
            '反应总时长(秒)': total_reaction,
            '总累积供氧(升)': total_cumulative
        })
        
        # 创建DataFrame并写入Excel
        ignition_df = pd.DataFrame(ignition_test_data)
        ignition_df.to_excel(writer, sheet_name='点火测试记录数据', index=False)
        
        # 设置点火测试记录数据的格式
        ignition_sheet = writer.sheets['点火测试记录数据']
        ignition_sheet.set_column('A:A', 20)
        ignition_sheet.set_column('B:G', 25)
        
        # 10.4) 平稳性分析
        print("\n=== 平稳性分析 ===")
        print(f"活跃传感器数量: {len(active_sensors)}")
        
        # 确定平稳期边界（传入反应结束时间）
        start_time, end_time = find_stable_period(df, cumulative_flow, REACTION_END_TIME)
        
        if start_time is not None and end_time is not None:
            print(f"平稳期起点: {start_time:.1f}秒 (总产氧量达到20L)")
            print(f"平稳期终点: {end_time:.1f}秒 (反应结束前120秒)")
            
            # 提取平稳期数据
            stable_data = extract_stable_data(df, start_time, end_time)
            
            # 线性拟合
            fit_params = linear_fit(stable_data)
            print(f"拟合斜率: {fit_params['slope']:.6f} L/min/s")
            print(f"拟合优度R²: {fit_params['r_squared']:.4f}")
            
            # 计算偏差
            deviations, y_fitted = calculate_deviations(stable_data, fit_params)
            
            # 计算稳定性指标
            metrics = calculate_stability_metrics(stable_data, deviations, cumulative_flow, df, active_sensors)
            
            # 滑动窗口分析
            window_results = sliding_window_analysis(stable_data)
            
            # 创建平稳性分析数据
            stability_data = []
            
            # 准备格式化的数据
            stability_data.append(['平稳性分析报告', '', '', ''])
            stability_data.append(['', '', '', ''])
            stability_data.append(['一、平稳期边界信息', '', '', ''])
            stability_data.append(['起始时间(s)', round(start_time, 1), '', ''])
            stability_data.append(['结束时间(s)', round(end_time, 1), '', ''])
            stability_data.append(['持续时间(s)', round(end_time - start_time, 1), '', ''])
            stability_data.append(['起始累积总流量(L)', round(metrics['start_cumul'], 2), '', ''])
            stability_data.append(['结束累积总流量(L)', round(metrics['end_cumul'], 2), '', ''])
            stability_data.append(['', '', '', ''])
            
            stability_data.append(['二、线性拟合参数', '', '', ''])
            stability_data.append(['拟合斜率(L/min/s)', round(fit_params['slope'], 6), '', ''])
            stability_data.append(['', '', '', '表示平稳期内流量的变化趋势。接近0表示流量稳定；负值表示流量缓慢下降；正值表示流量缓慢上升'])
            stability_data.append(['拟合截距(L/min)', round(fit_params['intercept'], 4), '', ''])
            stability_data.append(['拟合优度R²', round(fit_params['r_squared'], 4), '', ''])
            stability_data.append(['', '', '', '衡量数据与拟合直线的吻合程度(0-1)。越接近1表示数据越接近线性变化，流量波动越规律'])
            stability_data.append(['', '', '', ''])
            
            stability_data.append(['三、稳定性指标', '', '', ''])
            cv_eval = evaluate_metric(metrics['cv'], [0.05, 0.10, 0.15])
            stability_data.append(['相对基准线变异系数', round(metrics['cv'], 4), cv_eval, ''])
            stability_data.append(['', '', '', '实际流量偏离拟合直线的相对离散程度。反映流量围绕趋势线的波动大小'])
            stability_data.append(['最大偏差幅度(L/min)', round(metrics['max_deviation'], 4), '', ''])
            stability_data.append(['', '', '', '平稳期内实际流量与拟合值的最大偏差，反映流量波动的极值'])
            stability_data.append(['超调时间占比(%)', round(metrics['exceed_ratio'], 2), f'(阈值: ±{round(metrics["exceed_threshold"], 3)} L/min)', ''])
            stability_data.append(['', '', '', '偏差超过平均流量15%的时间占比。反映大幅波动的频繁程度'])
            stability_data.append(['平稳期平均流量(L/min)', round(metrics['mean_flow'], 3), '', ''])
            stability_data.append(['平稳期产氧效率(L/min)', round(metrics['efficiency'], 3), '(基于总流量)', ''])
            stability_data.append(['启动响应时间(s)', round(metrics['start_time_20L'], 1), '(总产氧量达到20L)', ''])
            stability_data.append(['', '', '', ''])
            
            # 转换为DataFrame
            stability_df = pd.DataFrame(stability_data, columns=['指标', '数值', '评价', '备注'])
            stability_df.to_excel(writer, sheet_name='平稳性分析', index=False, header=False)
            
            # 获取worksheet对象
            stability_sheet = writer.sheets['平稳性分析']
            
            # 设置列宽
            stability_sheet.set_column('A:A', 30)
            stability_sheet.set_column('B:B', 15)
            stability_sheet.set_column('C:C', 20)
            stability_sheet.set_column('D:D', 80)  # 增加备注列宽度以容纳说明文字
            
            # 定义格式
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            subtitle_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#D3D3D3'
            })
            
            explanation_format = workbook.add_format({
                'italic': True,
                'font_size': 10,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#F0F0F0'
            })
            
            # 合并标题单元格
            stability_sheet.merge_range('A1:D1', '平稳性分析报告', title_format)
            
            
            # 滑动窗口分析结果
            if window_results:
                # 添加滑动窗口分析说明
                current_row = 28  # 调整起始行
                stability_sheet.write(current_row, 0, '四、滑动窗口分析', subtitle_format)
                current_row += 1
                stability_sheet.write(current_row, 0, '说明：', workbook.add_format({'bold': True}))
                stability_sheet.merge_range(current_row, 1, current_row, 3, 
                                          '将平稳期数据按60秒窗口、10秒步长进行滑动分析，评估局部稳定性的时间演变', 
                                          explanation_format)
                current_row += 1
                
                # 添加稳定性评价标准说明
                stability_sheet.write(current_row, 0, '稳定性评价标准：', workbook.add_format({'bold': True}))
                current_row += 1
                stability_sheet.merge_range(current_row, 0, current_row, 3,
                                          '优秀: CV<1%  |  良好: 1%≤CV<2%  |  一般: 2%≤CV<4%  |  较差: CV≥4%',
                                          workbook.add_format({'align': 'center', 'italic': True, 'fg_color': '#F0F0F0'}))
                current_row += 2
                
                # 准备滑动窗口数据
                window_df = pd.DataFrame(window_results[:10])  # 最多显示10个窗口
                # 写入滑动窗口分析
                window_df.to_excel(writer, sheet_name='平稳性分析', startrow=current_row, index=False)
            
            # 创建图表数据
            # 准备完整的数据（每秒一个点）
            max_time = int(REACTION_END_TIME)  # 使用反应结束时间
            time_points = np.arange(0, max_time + 1, 1)
            
            # 创建图表数据
            chart_data = pd.DataFrame({'时间(s)': time_points})
            
            # 插值获取每秒的流量值
            chart_data['平均流量L/Min'] = np.interp(time_points, df['时间(s)'].values, df['平均流量L/Min'].values)
            chart_data['基准流量L/Min'] = np.interp(time_points, df['时间(s)'].values, df['基准流量L/Min'].values)
            
            # 为不同时段创建分段数据
            # 平稳期前的数据
            chart_data['平均流量_前'] = chart_data['平均流量L/Min'].copy()
            chart_data['基准流量_前'] = chart_data['基准流量L/Min'].copy()
            chart_data.loc[chart_data['时间(s)'] >= start_time, '平均流量_前'] = np.nan
            chart_data.loc[chart_data['时间(s)'] >= start_time, '基准流量_前'] = np.nan
            
            # 平稳期的数据
            chart_data['平均流量_稳定'] = np.nan
            chart_data['基准流量_稳定'] = np.nan
            chart_data['拟合流量_稳定'] = np.nan
            stable_mask = (chart_data['时间(s)'] >= start_time) & (chart_data['时间(s)'] <= end_time)
            chart_data.loc[stable_mask, '平均流量_稳定'] = chart_data.loc[stable_mask, '平均流量L/Min']
            chart_data.loc[stable_mask, '基准流量_稳定'] = chart_data.loc[stable_mask, '基准流量L/Min']
            chart_data.loc[stable_mask, '拟合流量_稳定'] = (
                fit_params['slope'] * chart_data.loc[stable_mask, '时间(s)'] + fit_params['intercept']
            )
            
            # 平稳期后的数据
            chart_data['平均流量_后'] = chart_data['平均流量L/Min'].copy()
            chart_data['基准流量_后'] = chart_data['基准流量L/Min'].copy()
            chart_data.loc[chart_data['时间(s)'] <= end_time, '平均流量_后'] = np.nan
            chart_data.loc[chart_data['时间(s)'] <= end_time, '基准流量_后'] = np.nan
            
            # 将数据写入Sheet的底部
            chart_start_row = 45  # 调整图表数据起始行
            
            # 在图表数据上方添加说明
            stability_sheet.write(chart_start_row - 2, 0, 
                                 '图表数据说明：灰色线条表示非平稳期，彩色线条表示平稳期', 
                                 workbook.add_format({'italic': True}))
            
            # 写入数据
            output_columns = ['时间(s)', '平均流量L/Min', '基准流量L/Min',
                            '平均流量_前', '基准流量_前',
                            '平均流量_稳定', '基准流量_稳定', '拟合流量_稳定',
                            '平均流量_后', '基准流量_后']
            chart_data[output_columns].to_excel(writer, sheet_name='平稳性分析', 
                                              startrow=chart_start_row, startcol=0, index=False)
            
            # 创建图表
            chart = workbook.add_chart({'type': 'line'})
            
            # 数据行数（包括标题行）
            data_rows = len(chart_data) + 1
            
            # 2. 平稳期的彩色曲线 - 先添加这些需要显示图例的系列
            # 平均流量 - 平稳期
            chart.add_series({
                'name': '平均流量',
                'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                'values': ['平稳性分析', chart_start_row + 1, 5, chart_start_row + data_rows - 1, 5],
                'line': {'color': '#0066CC', 'width': 3}
            })
            
            # 基准流量 - 平稳期
            chart.add_series({
                'name': '基准流量',
                'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                'values': ['平稳性分析', chart_start_row + 1, 6, chart_start_row + data_rows - 1, 6],
                'line': {'color': '#CC0000', 'width': 3, 'dash_type': 'dash'}
            })
            
            # 拟合直线 - 平稳期
            chart.add_series({
                'name': '拟合直线',
                'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                'values': ['平稳性分析', chart_start_row + 1, 7, chart_start_row + data_rows - 1, 7],
                'line': {'color': '#00CC00', 'width': 2.5}
            })
            
            # 1. 平稳期前的灰色曲线（如果存在）- 后添加，不设置名称
            if start_time > 0:
                # 平均流量 - 平稳期前
                chart.add_series({
                    'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                    'values': ['平稳性分析', chart_start_row + 1, 3, chart_start_row + data_rows - 1, 3],
                    'line': {'color': '#B0B0B0', 'width': 2},
                    'marker': {'type': 'none'}
                })
                
                # 基准流量 - 平稳期前
                chart.add_series({
                    'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                    'values': ['平稳性分析', chart_start_row + 1, 4, chart_start_row + data_rows - 1, 4],
                    'line': {'color': '#B0B0B0', 'width': 2, 'dash_type': 'dash'},
                    'marker': {'type': 'none'}
                })
            
            # 3. 平稳期后的灰色曲线（如果存在）- 后添加，不设置名称
            if end_time < REACTION_END_TIME:
                # 平均流量 - 平稳期后
                chart.add_series({
                    'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                    'values': ['平稳性分析', chart_start_row + 1, 8, chart_start_row + data_rows - 1, 8],
                    'line': {'color': '#B0B0B0', 'width': 2},
                    'marker': {'type': 'none'}
                })
                
                # 基准流量 - 平稳期后
                chart.add_series({
                    'categories': ['平稳性分析', chart_start_row + 1, 0, chart_start_row + data_rows - 1, 0],
                    'values': ['平稳性分析', chart_start_row + 1, 9, chart_start_row + data_rows - 1, 9],
                    'line': {'color': '#B0B0B0', 'width': 2, 'dash_type': 'dash'},
                    'marker': {'type': 'none'}
                })
            
            # 设置图表属性
            chart.set_title({'name': f'流量曲线全程分析（平稳期：{int(start_time)}-{int(end_time)}秒）'})
            chart.set_x_axis({
                'name': '时间 (秒)', 
                'min': 0, 
                'max': REACTION_END_TIME,
                'major_unit': max(100, int(REACTION_END_TIME/10))  # 动态设置主刻度间隔
            })
            chart.set_y_axis({'name': '流量 (L/Min)', 'min': 0})
            chart.set_size({'width': 750, 'height': 450})
            
            # 设置图例，只显示前3个系列
            chart.set_legend({
                'position': 'right',
                'delete_series': [3, 4, 5, 6] if start_time > 0 and end_time < REACTION_END_TIME else
                                 [3, 4] if start_time > 0 else
                                 [3, 4] if end_time < REACTION_END_TIME else
                                 []
            })
            
            # 插入图表
            stability_sheet.insert_chart('F2', chart)
            
            print("\n✅ 平稳性分析完成")
        else:
            print("\n❌ 无法进行平稳性分析")
        
        # 10.5) 平均异常分析表（0.9阈值）
        # 窗口时间：从反应开始7分钟后到反应结束前2分钟
        window_start = 7 * 60  # 7分钟 = 420秒
        window_end = REACTION_END_TIME - 2 * 60  # 使用统一定义的反应结束时间
        
        print(f"\n=== 0.9阈值异常分析 ===")
        print(f"数据采集时间: 0秒 到 {df['时间(s)'].iloc[-1]:.1f}秒")
        print(f"反应结束时间: {REACTION_END_TIME:.1f}秒 ({REACTION_END_TIME/60:.2f}分钟)")
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
        
        # 10.6) 综合异常分析（合并所有传感器和平均值）
        print("\n=== 综合异常分析 ===")
        
        # 收集所有异常分析数据
        all_violations_data = []
        sheet_row_positions = {}  # 记录每个分析的起始行位置
        current_row = 0
        
        # 分析各传感器
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
                # 记录这个传感器的数据起始位置
                sheet_row_positions[sensor_name] = current_row
                all_violations_data.append({
                    'title': f'{sensor_name}异常分析',
                    'data': pd.DataFrame(vio_records)
                })
                current_row += len(vio_records) + 3  # 数据行数 + 标题行 + 空行
        
        # 分析平均值
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
            sheet_row_positions['平均值'] = current_row
            all_violations_data.append({
                'title': '平均值异常分析',
                'data': pd.DataFrame(vio_records)
            })
        
        # 将所有异常分析写入一个sheet
        if all_violations_data:
            # 创建一个新的worksheet
            worksheet_violations = workbook.add_worksheet('综合异常分析')
            writer.sheets['综合异常分析'] = worksheet_violations
            
            # 定义格式
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#F0F0F0',
                'border': 1,
                'text_wrap': True,
                'valign': 'vcenter',
                'align': 'center'
            })
            
            data_format = workbook.add_format({
                'border': 1,
                'align': 'center'
            })
            
            # 写入数据
            row_offset = 0
            for item in all_violations_data:
                # 写入标题
                worksheet_violations.merge_range(row_offset, 0, row_offset, 5, 
                                               item['title'], title_format)
                row_offset += 1
                
                # 写入表头
                headers = ['开始时间(秒)', '结束时间(秒)', '持续时间(秒)', 
                          '流量差异(升)', '起点累积流量(升)', '终点累积流量(升)']
                for col, header in enumerate(headers):
                    worksheet_violations.write(row_offset, col, header, header_format)
                row_offset += 1
                
                # 写入数据
                for idx, row in item['data'].iterrows():
                    for col, header in enumerate(headers):
                        worksheet_violations.write(row_offset, col, row[header], data_format)
                    row_offset += 1
                
                # 添加空行
                row_offset += 2
            
            # 设置列宽
            worksheet_violations.set_column('A:F', 18)
            
            print("✅ 综合异常分析表创建完成")
        
        # 10.7) 燃烧定位分析（合并关键时间点和完整数据）
        if not scale_df.empty:
            print("\n=== 燃烧定位分析 ===")
            
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
            
            # 创建关键时间点数据
            key_times = [0, 60, 120, 180, 240, 300, 360, 600, 900, 1200]
            # 只保留在反应时间范围内的关键时间点
            key_times = [t for t in key_times if t <= REACTION_END_TIME]
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
            
            # 创建燃烧定位综合sheet
            worksheet_burn = workbook.add_worksheet('燃烧定位分析')
            writer.sheets['燃烧定位分析'] = worksheet_burn
            
            # 定义格式
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#4472C4',
                'font_color': 'white'
            })
            
            subtitle_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#F0F0F0',
                'border': 1,
                'text_wrap': True,
                'valign': 'vcenter',
                'align': 'center'
            })
            
            data_format = workbook.add_format({
                'border': 1,
                'align': 'center'
            })
            
            # 写入总标题
            worksheet_burn.merge_range('A1:D1', '燃烧定位分析报告', title_format)
            
            # 1. 关键时间点燃烧定位表
            worksheet_burn.merge_range('A3:D3', '关键时间点燃烧定位', subtitle_format)
            
            # 写入关键时间点数据
            key_df = pd.DataFrame(key_burn_data)
            headers = ['时间(s)', '产氧量(L)', '燃烧深度(mm)', '大致位置']
            
            # 写入表头
            for col, header in enumerate(headers):
                worksheet_burn.write(3, col, header, header_format)
            
            # 写入数据
            for idx, row in key_df.iterrows():
                for col, header in enumerate(headers):
                    worksheet_burn.write(4 + idx, col, row[header], data_format)
            
            # 2. 完整燃烧定位数据（每10秒显示一次，最多显示30行）
            start_row = 4 + len(key_df) + 3  # 留出空行
            worksheet_burn.merge_range(start_row - 1, 0, start_row - 1, 3, 
                                      '燃烧定位详细数据（每10秒采样）', subtitle_format)
            
            # 创建采样数据（每10秒一个点）
            sampled_burn_data = []
            for i in range(0, len(burn_data), 20):  # 每10秒取一个点（0.5秒采样）
                sampled_burn_data.append(burn_data[i])
                if len(sampled_burn_data) >= 30:  # 最多显示30行
                    break
            
            # 写入表头
            for col, header in enumerate(headers):
                worksheet_burn.write(start_row, col, header, header_format)
            
            # 写入采样数据
            for idx, data in enumerate(sampled_burn_data):
                for col, header in enumerate(headers):
                    worksheet_burn.write(start_row + 1 + idx, col, data[header], data_format)
            
            # 设置列宽
            worksheet_burn.set_column('A:A', 12)
            worksheet_burn.set_column('B:B', 12)
            worksheet_burn.set_column('C:C', 15)
            worksheet_burn.set_column('D:D', 15)
            
            # 如果还有更多数据，添加说明
            if len(burn_data) > 600:  # 如果数据超过300秒
                note_row = start_row + len(sampled_burn_data) + 3
                worksheet_burn.merge_range(note_row, 0, note_row, 3, 
                                          f'注：完整数据共{len(burn_data)}个点，此处仅显示前{len(sampled_burn_data)}个采样点', 
                                          workbook.add_format({'italic': True, 'align': 'center'}))
            
            print("✅ 燃烧定位分析表创建完成")
        
        # 10.8) 创建曲线图Sheet - 显示所有传感器流量曲线
        # 为图表数据创建每秒一个点的数据（用于图表显示）
        # 只选择整数秒的数据点
        chart_data = df[df['时间(s)'] % 1 == 0].copy()
        
        # 如果没有整数秒的数据，则取最接近整数秒的数据
        if len(chart_data) == 0:
            # 创建整数秒的时间点
            max_time = int(REACTION_END_TIME)  # 使用反应结束时间
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
        chart1.set_x_axis({
            'name': '时间 (秒)',
            'min': 0,
            'max': REACTION_END_TIME,
            'major_unit': max(100, int(REACTION_END_TIME/10))
        })
        chart1.set_y_axis({'name': '流量 (L/Min)'})
        chart1.set_size({'width': 720, 'height': 480})
        worksheet.insert_chart('H2', chart1)
        
        # 10.9) 创建流量-温度综合分析图（如果有温度数据）
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
                chart3.set_x_axis({
                    'name': '时间 (秒)',
                    'min': 0,
                    'max': REACTION_END_TIME,
                    'major_unit': max(100, int(REACTION_END_TIME/10))
                })
                chart3.set_y_axis({'name': '流量 (L/Min)', 'major_gridlines': {'visible': True}})
                chart3.set_y2_axis({'name': '温度 (°C)'})
                chart3.set_size({'width': 720, 'height': 480})
                worksheet_temp.insert_chart('G2', chart3)
                
                print("✅ 流量-温度综合分析图创建成功")
            else:
                print("⚠️ 数据不足，无法创建流量-温度综合分析图")
        else:
            print("⚠️ 无温度数据，跳过流量-温度综合分析图")
        
        # 10.10) 产氧-深度曲线
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
            chart2.set_x_axis({
                'name': '时间 (秒)',
                'min': 0,
                'max': REACTION_END_TIME,
                'major_unit': max(100, int(REACTION_END_TIME/10))
            })
            chart2.set_y_axis({'name': '产氧量 (L)', 'major_gridlines': {'visible': True}})
            chart2.set_y2_axis({'name': '燃烧深度 (mm)'})
            chart2.set_size({'width': 720, 'height': 480})
            worksheet_depth.insert_chart('E2', chart2)
    
    print(f"✅ 生成报告: {output_file}")

print(f"\n{'='*50}")
print(f"所有报告生成完成！")
print(f"输出目录: {OUT_DIR}")
print(f"{'='*50}")