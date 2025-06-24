# 报告生成器（Cross-Platform Report Generator）

## **更新历史**

### 2024-06-24

- 新增功能：

  增加了“平均异常分析表（0.9阈值）”，用于对氧烛燃烧期间（从第7分钟到燃尽前2分钟）各通道平均流量是否低于基准流量的90%进行统计和分析。该表独立于原有性能指标分析，仅作为额外的质量分析参考，不影响原有的达标率、性能评判标准。

- **算法优化**：

  优化了异常分析时间窗口的计算方式。窗口结束时间不再是总时长减120秒，而是根据“最后一次平均流量大于0的时间点”作为燃尽点，再倒推2分钟确定异常分析的结束点，更准确反映实际燃烧终止位置。

- **补丁兼容**：

  代码改动以补丁（patch）方式维护，支持在现有项目上便捷更新，无需手动插入代码。便于团队成员保持脚本一致性。

- **代码结构保持不变**：

  所有新功能均以最小侵入方式插入，未改变原有表格、分析流程和主输出结构，保证项目历史数据和新功能兼容。

------

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