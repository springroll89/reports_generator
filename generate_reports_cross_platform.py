#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ————————————
# 一、配置路径
# ————————————

import sys

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
# 三、读取基准曲线
# ————————————
baseline_csv = os.path.join(INPUT_DIR, '.基准曲线 1.csv')
base_df = pd.read_csv(baseline_csv)
base_t, base_f = base_df['time'].values, base_df['flow'].values

# ————————————
# 四、读取刻度表
# ————————————
scale_file = os.path.join(INPUT_DIR, '.副本4P、2P22M氧烛分层及刻度.xlsx')
scale_4p22m = pd.read_excel(scale_file, sheet_name='4P22M')
scale_2p22m = pd.read_excel(scale_file, sheet_name='2P22M')

# ————————————
# 五、报告通用参数
# ————————————
KEY_TIMES = [60, 100, 142, 360, 1200]   # 关键时间点（秒）
NOTE_TEXT = (
    "\"流量差异\"指在不达标时间段内，实际瞬时流量低于基准曲线的累积体积差值（L）。\n"
    "计算方法：对(b(t)-a(t))积分并除以60。\n"
    "\"起点累积流量(升)\"指从点火时刻到异常起始时刻所累积的总产氧量（L）。\n"
    "\"终点累积流量(升)\"指从点火时刻到异常结束时刻所累积的总产氧量（L）。"
)

# ————————————
# 六、准备输出目录
# ————————————
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# ————————————
# 七、扫描并处理每个原始表
# ————————————
pattern = os.path.join(INPUT_DIR, '副本氧气流量采集分析表*.xlsx')
files = glob.glob(pattern)
if not files:
    print(f"❌ 在 {INPUT_DIR} 未找到任何符合 '副本氧气流量采集分析表*.xlsx' 的文件")
    exit(1)

for path in files:
    label = os.path.splitext(os.path.basename(path))[0]
    # 1) 读取原始（已预处理）Excel
    raw_df = pd.read_excel(path)

    # 2) 解析采集时间，并标准化成 秒 数
    raw_df['采集时间'] = pd.to_datetime(
        raw_df['采集时间'], format='%Y年%m月%d日%H:%M:%S'
    )
    t0 = raw_df['采集时间'].iloc[0]
    raw_df['时间(s)'] = (raw_df['采集时间'] - t0).dt.total_seconds()

    # 3) 计算平均与基准
    SENSORS = [f"{i}号瞬时流量L/Min" for i in range(1,5)]
    df = raw_df.copy()
    
    # 判断是4P还是2P模式
    # 通过检查3号和4号传感器是否有非零数据
    sensor_3_4_has_data = (df['3号瞬时流量L/Min'] > 0).any() or (df['4号瞬时流量L/Min'] > 0).any()
    
    if sensor_3_4_has_data:
        # 4P模式：使用所有4个传感器
        active_sensors = SENSORS
        scale_df = scale_4p22m
        mode = '4P22M'
    else:
        # 2P模式：只使用1号和2号传感器
        active_sensors = SENSORS[:2]
        scale_df = scale_2p22m
        mode = '2P22M'
    
    df['平均流量L/Min'] = df[active_sensors].mean(axis=1)
    df['基准流量L/Min'] = np.interp(df['时间(s)'], base_t, base_f)

    # 4) 原始数据 Sheet
    raw_cols = ['时间(s)'] + SENSORS + ['平均流量L/Min', '基准流量L/Min']
    raw_data = df[raw_cols]

    # 5) 关键点分析
    crit = []
    for t in KEY_TIMES:
        sub = df[df['时间(s)'] <= t]
        rec = {'时间点(s)': t}
        for s in SENSORS:
            rec[f'{s}_总流量(L)'] = round(np.trapz(sub[s], sub['时间(s)'])/60, 2)
        rec['平均总流量(L)'] = round(np.trapz(sub['平均流量L/Min'], sub['时间(s)'])/60, 2)
        xi = np.linspace(0, t, int(t*2)+1)
        rec['基准总流量(L)'] = round(np.trapz(np.interp(xi, base_t, base_f), xi)/60, 2)
        crit.append(rec)
    crit_df = pd.DataFrame(crit)

    # 6) 性能指标分析（含产氧时间）
    t_arr = df['时间(s)'].values
    base_curve = np.interp(t_arr, base_t, base_f)
    perf = []
    for idx, s in enumerate(SENSORS, start=1):
        arr = df[s].values
        # 启动时长
        pos_idx = np.where(arr > 0)[0]
        start   = df['时间(s)'].iloc[pos_idx[0]] if len(pos_idx) else np.nan
        # 达峰时长
        peak_idx = np.where(arr >= 3.75)[0]
        peak     = df['时间(s)'].iloc[peak_idx[0]] if len(peak_idx) else np.nan
        # 累计流量
        total    = round(np.trapz(arr, t_arr)/60, 2)
        # 达标率
        rate     = round((arr >= base_curve).mean()*100, 2)
        # 产氧时间：从 start 到最后一次 >0
        last_idx = pos_idx[-1] if len(pos_idx) else None
        end_time = df['时间(s)'].iloc[last_idx] if last_idx is not None else np.nan
        oxy_time = round((end_time - start)/60, 2) if last_idx is not None else np.nan

        perf.append({
            '设备':            f'{idx}号',
            '启动时长(秒)':    round(start, 2),
            '达峰时长(秒)':    round(peak, 2),
            '累计流量(升)':    total,
            '达标率(%)':      rate,
            '产氧时间(分钟)': oxy_time
        })

    # 平均值
    arr = df['平均流量L/Min'].values
    pos_idx = np.where(arr > 0)[0]
    start2   = df['时间(s)'].iloc[pos_idx[0]] if len(pos_idx) else np.nan
    peak2_idx= np.where(arr >= 3.75)[0]
    peak2    = df['时间(s)'].iloc[peak2_idx[0]] if len(peak2_idx) else np.nan
    total2   = round(np.trapz(arr, t_arr)/60, 2)
    rate2    = round((arr >= base_curve).mean()*100, 2)
    last_idx2= pos_idx[-1] if len(pos_idx) else None
    end2     = df['时间(s)'].iloc[last_idx2] if last_idx2 is not None else np.nan
    oxy2     = round((end2 - start2)/60, 2) if last_idx2 is not None else np.nan

    perf.append({
        '设备':            '平均值',
        '启动时长(秒)':    round(start2, 2),
        '达峰时长(秒)':    round(peak2, 2),
        '累计流量(升)':    total2,
        '达标率(%)':      rate2,
        '产氧时间(分钟)': oxy2
    })
    perf_df = pd.DataFrame(perf)

    # 7) 异常分析（同原脚本）
    ab_sheets = {}
    for idx, name in enumerate([f'{i}号' for i in range(1,5)] + ['平均值']):
        arr = df[SENSORS[idx]].values if idx < 4 else df['平均流量L/Min'].values
        recs, st, diff = [], None, 0
        for t, a, b in zip(df['时间(s)'], arr, base_curve):
            if a < b:
                if st is None: st = t
                diff += (b - a)/60
            else:
                if st is not None:
                    cum_s = round(np.trapz(
                        df.loc[df['时间(s)'] <= st, active_sensors].sum(axis=1),
                        df.loc[df['时间(s)'] <= st, '时间(s)'])/60, 2)
                    cum_e = round(np.trapz(
                        df.loc[df['时间(s)'] <= t, active_sensors].sum(axis=1),
                        df.loc[df['时间(s)'] <= t, '时间(s)'])/60, 2)
                    recs.append({
                        '开始时间(秒)':      round(st, 2),
                        '结束时间(秒)':      round(t, 2),
                        '持续时间(秒)':      round(t - st, 2),
                        '流量差异(升)':      round(diff, 2),
                        '起点累积流量(升)': cum_s,
                        '终点累积流量(升)': cum_e
                    })
                    st, diff = None, 0
        if st is not None:
            end = df['时间(s)'].iloc[-1]
            cum_s = round(np.trapz(
                df.loc[df['时间(s)'] <= st, active_sensors].sum(axis=1),
                df.loc[df['时间(s)'] <= st, '时间(s)'])/60, 2)
            cum_e = round(np.trapz(
                df.loc[df['时间(s)'] <= end, active_sensors].sum(axis=1),
                df.loc[df['时间(s)'] <= end, '时间(s)'])/60, 2)
            recs.append({
                '开始时间(秒)':      round(st, 2),
                '结束时间(秒)':      round(end, 2),
                '持续时间(秒)':      round(end - st, 2),
                '流量差异(升)':      round(diff, 2),
                '起点累积流量(升)': cum_s,
                '终点累积流量(升)': cum_e
            })
        ab_sheets[name] = pd.DataFrame(recs)

    # ========== 新增 平均异常分析（绝对0.9L/min阈值） ==========
    # 窗口：7min ~ 燃尽前2min
    win_start = 7 * 60
    avg_arr   = df['平均流量L/Min'].values
    ts        = df['时间(s)'].values

    # 找到最后一个非零平均流量时刻作为燃尽点
    nz = np.where(avg_arr > 0)[0]
    burn_out_time = ts[nz[-1]] if nz.size else ts[0]
    win_end = max(burn_out_time - 120, win_start)

    # 异常判据：平均流量 < 0.9L/min（绝对值！）
    mask = (ts >= win_start) & (ts <= win_end)
    abnormal = (avg_arr < 0.9) & mask

    exception_records = []
    in_block = False
    for i in range(len(ts)):
        if abnormal[i]:
            if not in_block:
                block_start = ts[i]
                start_oxy = np.trapz(avg_arr[:i+1], ts[:i+1]) / 60
                in_block = True
            block_end = ts[i]
            end_oxy = np.trapz(avg_arr[:i+1], ts[:i+1]) / 60
        else:
            if in_block:
                exception_records.append({
                    "开始时间(秒)": round(block_start, 2),
                    "结束时间(秒)": round(block_end, 2),
                    "持续时间(秒)": round(block_end - block_start, 2),
                    "起点累积流量(升)": round(start_oxy, 2),
                    "终点累积流量(升)": round(end_oxy, 2),
                })
                in_block = False
    if in_block:
        exception_records.append({
            "开始时间(秒)": round(block_start, 2),
            "结束时间(秒)": round(block_end, 2),
            "持续时间(秒)": round(block_end - block_start, 2),
            "起点累积流量(升)": round(start_oxy, 2),
            "终点累积流量(升)": round(end_oxy, 2),
        })
    df_avg_exception = pd.DataFrame(exception_records)

    # 8) 燃烧定位表计算
    # 计算每个时间点的累积产氧量
    cumulative_oxygen = []
    for i in range(len(df)):
        sub_df = df.iloc[:i+1]
        total_oxygen = np.trapz(sub_df[active_sensors].sum(axis=1), sub_df['时间(s)']) / 60
        cumulative_oxygen.append(total_oxygen)
    
    # 计算总产氧量
    total_oxygen_production = cumulative_oxygen[-1]
    
    # 根据刻度表计算燃烧深度
    burn_location = []
    for cum_oxy in cumulative_oxygen:
        if total_oxygen_production > 0:
            # 计算当前累积产氧量占总产氧量的百分比
            percentage = cum_oxy / total_oxygen_production
            
            # 在刻度表中查找对应的燃烧深度
            if percentage <= 0:
                depth = 0
                location = "未开始"
            elif percentage >= 1:
                depth = scale_df['刻度值/mm'].max()
                location = scale_df.iloc[-1]['大致位置']
            else:
                # 插值计算深度
                depth = np.interp(percentage, 
                                scale_df['有效燃烧百分比'], 
                                scale_df['刻度值/mm'])
                # 查找对应位置
                idx = np.searchsorted(scale_df['有效燃烧百分比'], percentage)
                if idx >= len(scale_df):
                    location = scale_df.iloc[-1]['大致位置']
                else:
                    location = scale_df.iloc[idx]['大致位置']
        else:
            depth = 0
            location = "未开始"
        
        burn_location.append({
            '时间(s)': df['时间(s)'].iloc[len(burn_location)],
            '产氧量(L)': round(cum_oxy, 2),
            '燃烧深度(mm)': round(depth, 2),
            '大致位置': location
        })
    
    burn_location_df = pd.DataFrame(burn_location)
    
    # 选择关键时间点的燃烧定位信息
    key_burn_location = []
    for t in [0, 60, 120, 300, 600, 900, 1200]:
        if t <= df['时间(s)'].max():
            idx = (df['时间(s)'] - t).abs().argmin()
            key_burn_location.append(burn_location[idx])
    key_burn_location_df = pd.DataFrame(key_burn_location)

    # 9) 写入 Excel & 插入曲线图
    out_file = os.path.join(OUT_DIR, f"{label}_分析报告.xlsx")
    with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
        raw_data.to_excel(writer, '原始数据', index=False)
        crit_df.to_excel(writer, '关键点分析', index=False)
        perf_df.to_excel(writer, '性能指标分析', index=False)
        key_burn_location_df.to_excel(writer, '燃烧定位表', index=False)
        # ...已有写入...
        df_avg_exception.to_excel(writer, '平均异常分析表_0.9阈值', index=False) 

        for name, df_ab in ab_sheets.items():
            sheet = f'异常分析_{name}'
            df_ab.to_excel(writer, sheet, index=False)
            ws = writer.sheets[sheet]
            ws.write(len(df_ab) + 2, 0, NOTE_TEXT)

        wb       = writer.book
        
        # 原始曲线图
        chart_ws = wb.add_worksheet('曲线图')
        chart    = wb.add_chart({'type': 'line'})
        n        = len(raw_data)
        for i in range(1, len(raw_cols)):
            chart.add_series({
                'name':       ['原始数据', 0, i],
                'categories': ['原始数据', 1, 0, n, 0],
                'values':     ['原始数据', 1, i, n, i],
            })
        chart.set_x_axis({'name': '时间(s)'})
        chart.set_y_axis({'name': '流量(L/Min)'})
        chart.set_title({'name': '流量时序曲线'})
        chart_ws.insert_chart('B2', chart, {'x_scale': 1.2, 'y_scale': 1.2})
        
        # 产氧-深度双坐标曲线图
        depth_ws = wb.add_worksheet('产氧-深度曲线')
        
        # 为产氧-深度曲线准备数据
        burn_location_df.to_excel(writer, '燃烧定位数据', index=False)
        
        # 创建双坐标图
        combo_chart = wb.add_chart({'type': 'line'})
        
        # 累积产氧量曲线（左Y轴）
        combo_chart.add_series({
            'name':       '累积产氧量',
            'categories': ['燃烧定位数据', 1, 0, len(burn_location_df), 0],
            'values':     ['燃烧定位数据', 1, 1, len(burn_location_df), 1],
            'line':       {'color': 'blue', 'width': 2},
        })
        
        # 燃烧深度曲线（右Y轴）
        combo_chart.add_series({
            'name':       '燃烧深度',
            'categories': ['燃烧定位数据', 1, 0, len(burn_location_df), 0],
            'values':     ['燃烧定位数据', 1, 2, len(burn_location_df), 2],
            'y2_axis':    True,
            'line':       {'color': 'red', 'width': 2},
        })
        
        combo_chart.set_x_axis({
            'name': '时间(s)',
            'min': 0,
            'max': 1200,
        })
        combo_chart.set_y_axis({
            'name': '累积产氧量(L)',
            'min': 0,
        })
        combo_chart.set_y2_axis({
            'name': '燃烧深度(mm)',
            'min': 0,
            'max': scale_df['刻度值/mm'].max(),
        })
        combo_chart.set_title({'name': f'产氧-深度曲线 ({mode})'})
        combo_chart.set_size({'width': 720, 'height': 480})
        
        depth_ws.insert_chart('B2', combo_chart)
        
        # 在第一个工作表中添加模式说明
        ws_info = writer.sheets['原始数据']
        ws_info.write(len(raw_data) + 2, 0, f'检测模式: {mode}')

print("✅ 全部报告已生成至:", OUT_DIR)
