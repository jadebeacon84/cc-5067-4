"""母带试听面板启动器（兼容隔离模式 -I）。

用法（仓库根目录）:
    /tmp/venv/bin/python -I scripts/panel.py
然后浏览器打开 http://127.0.0.1:7860 。

也可不加 -I 直接运行:
    /tmp/venv/bin/python -m dyneq.app
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyneq.app import main

if __name__ == "__main__":
    main()
