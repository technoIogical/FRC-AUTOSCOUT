*** Less complexity
1. only one main menu / view
2. Local DuckDB + Cloud Supabase (need to be better)
3. GUI: Flet
4. Ball / Robot Tracking: supervision
5. Musr use Log: Loguru
6. Data Analysis: Pandas / Polars + Pandas Profiling

# Functions
1. tracking robots to x and y in field map axis
2. tracking balls to x y z
3. counts how many balls the robot shoot

# Project Structure
```
FRC-AUTOSCOUT/
├── main.py                        # 入口，启动 Flet GUI
├── config.py                      # 全局配置 + loguru 初始化
├── requirements.txt               # Python 依赖
├── .gitignore
├── Architecture.md
├── resources/
│   ├── images/
│   │   └── field_top_view.png     # FRC 场地俯视图
│   └── models/
│       ├── robot.pt               # YOLO 机器人检测模型
│       └── ball.pt                # YOLO 球检测模型
├── data/                          # DuckDB 本地数据库文件
├── logs/                          # loguru 日志输出
└── src/
    ├── __init__.py
    ├── video_source.py            # 视频输入：摄像头 / 文件统一接口
    ├── detector.py                # YOLO 双模型检测封装
    ├── tracker.py                 # ByteTrack 多目标追踪 + Homography 坐标映射
    ├── shot_counter.py            # 投球事件检测与计数
    ├── db/
    │   ├── __init__.py
    │   ├── local_db.py            # DuckDB 本地存储
    │   └── cloud_db.py            # Supabase 云端同步
    ├── analysis/
    │   ├── __init__.py
    │   └── analyzer.py            # Polars 查询 + ydata-profiling 报告
    └── ui/
        ├── __init__.py
        └── main_view.py           # Flet 单页 GUI
```