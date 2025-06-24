# 报告生成器（Cross-Platform Report Generator）




## **更新历史**



### **2025-06-24**



- **新增功能**

  

  - 增加了“平均异常分析表_0.9阈值”Sheet：在Excel报告中自动新增一张表，专门用于统计“第7分钟（420秒）起至燃尽前2分钟”区间内，平均流量低于**绝对0.9L/min**的所有异常时段及其持续时间、起止时刻、累积产氧量等信息。
  - 表格字段包括：开始时间（秒）、结束时间（秒）、持续时间（秒）、起点累积流量（升）、终点累积流量（升）。

  

- **窗口区间优化**

  

  - 异常分析的考察时间段为“点火第7分钟起至燃尽前2分钟”，窗口右端点自动适配每次实验的真实燃尽时间，保证分析科学合理。

  

- **原有分析表**

  

  - 其余所有数据分析、性能指标和异常分析表均未变更，继续保留基准曲线相关判据。

  

- **BUG修复&体验优化**

  

  - 处理了异常分析时段切分准确性，累计流量计算更为严谨，逻辑更稳健。

  

---
## 项目简介

本项目为化学产氧器测试报告自动生成工具，支持 Windows 和 macOS 双平台。通过读取原始测量表、基准曲线和刻度表，自动生成包含数据分析、关键点提取、性能指标、异常分析和曲线图的完整 Excel 报告，以及对应的图表 PNG。可通过 Python 直接运行，也可使用 PyInstaller 打包为独立可执行文件。

## 目录结构

```
├── reports_input/                    # 输入数据目录
│   ├── 副本氧气流量采集分析表YYYYMMDD.xlsx  # 原始测量表（按前缀匹配）
│   ├── .基准曲线 1.csv                # 基准曲线数据
│   └── .副本4P、2P22M氧烛分层及刻度.xlsx  # 刻度表（4P22M/2P22M）
│
├── reports_output/                   # 输出报告目录（脚本运行自动创建）
│   └── 分析报告_YYYYMMDD_HHMMSS.xlsx    # 生成的完整报告
│
├── generate_reports_cross_platform.py # 跨平台主脚本
├── requirements.txt                  # Python 依赖列表
├── .github/
│   └── workflows/build.yml           # CI/CD 配置（GitHub Actions）
└── README.md                         # 本说明文件
```

## 环境要求

- Python 3.7+
- Windows 10 / macOS 10.15+
- 安装依赖：`pip install -r requirements.txt`

## 使用方法

1. 将原始测量表、基准曲线和刻度表放入 `reports_input/` 目录：
   - **原始测量表要求：**
     - 格式：质量流量仪导出的 `xls` 表格，另存为 `xlsx` 格式
     - 第二行必须是反应启动的时刻，末尾行无要求
   - **基准曲线：** `.基准曲线 1.csv`

- **刻度表：** `.副本4P、2P22M氧烛分层及刻度.xlsx`

1. 运行脚本：

   ```bash
   python generate_reports_cross_platform.py
   ```

2. 查看 `reports_output/` 目录，获取带有以下工作表的完整报告：

   - 原始数据
   - 关键点分析
   - 性能指标分析
   - 燃烧定位表
   - 异常分析_1号 … 异常分析_4号
   - 异常分析_平均值
   - 曲线图、产氧-深度曲线
   - 燃烧定位数据
   - 平均异常分析表_0.9阈值（2025-6-24更新加入）

## 打包为可执行文件

使用 PyInstaller 将脚本打包：

```bash
pyinstaller --onefile --name="报告生成器" generate_reports_cross_platform.py
```

打包完成后，可在 `dist/` 目录下获取 `报告生成器.exe`（Windows）或可执行文件（macOS）。

## CI/CD（GitHub Actions）

- Workflow 文件：`.github/workflows/build.yml`
- 每次推送（push）或发布（tag）时，自动安装依赖并执行 PyInstaller 打包，上传产物为 GitHub Release 附件。

## 贡献与许可证

欢迎提交 Issue 和 Pull Request。请遵循项目的编码规范和测试流程。

本项目采用 MIT 许可证，详情参见 LICENSE 文件。