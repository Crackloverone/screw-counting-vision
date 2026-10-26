# 螺丝计数系统

## 功能
- USB 相机连接
- 海康相机连接  
- 图片上传计数
- 模型选择

## 目录结构
```
newls/
├── app.py              # 主程序
├── requirements.txt    # 依赖
├── templates/
│   └── index.html    # 前端页面
├── models/            # 模型文件夹
├── data/             # 数据文件夹
└── uploads/          # 上传图片
```

## 运行
```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python app.py
```

访问 http://localhost:5000

## 使用
1. 选择数据源: USB相机/海康相机/上传图片
2. 选择本地模型
3. 点击计数