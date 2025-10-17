## 新澳门六合彩分析系统（2025）

- 爬虫：`kj.123720c.com/kj/?year=2025`（逐期抓取001–290）和`kj.123720c.com/kj/zl.html`（属性页）。
- 存储：SQLite（`data/lhc.sqlite3`）。
- 分析：频率统计、冷热、交替矩阵、回测、推荐；特码多因子（衰减频率+转移+共现+类别先验）。

### 安装
- Python 3.13+，然后执行：
```bash
pip install -r requirements.txt
```

### 使用
```bash
python3 -m lhc.cli initdb
python3 -m lhc.cli fetch-meta --year 2025
python3 -m lhc.cli fetch-draws --year 2025 --start 1 --end 290
python3 -m lhc.cli analyze --year 2025 --start 1 --end 289
python3 -m lhc.cli backtest --year 2025 --start 220 --end 290 --window 50
python3 -m lhc.cli special-backtest --year 2025 --start 220 --end 290 --window 60 --prev-k 3 --decay 0.97 --topk 5
```
