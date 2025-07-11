#!/bin/bash
# 如果你用的是虚拟环境，就先激活它（可选）
# source venv/bin/activate
# 切到脚本所在目录
cd /Users/chunjuan/Documents/reports_generator
# 执行报告生成脚本
python3 generate_reports_cross_platform.py
# 可选：脚本跑完后自动打开生成的压缩包
open reports.zip
