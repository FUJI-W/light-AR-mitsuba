# light-AR-mitsuba
> 本项目是 [light-AR](https://github.com/FUJI-W/light-AR) 项目的简化版，主要为了更方便地进行跨平台的构建。

- 渲染引擎由 [OptixRenderer](https://github.com/lzqsd/OptixRenderer) 转为 [Mitsuba](https://www.mitsuba-renderer.org/index_old.html) ，更好地支持Windows等平台；
- 优化了物体插入交互逻辑，去除了运行中间结果展示界面等。

### 图形界面

![image-20230401235650340](https://cdn.jsdelivr.net/gh/SnowOnVolcano/imagebed/202304012356620.png)

### 构建与运行

```python
# step 1. 下载 mitsuba 渲染程序
# step 2. 更改 config.py 中对应项
# step 3. 运行 app.py -> 示例命令 `python app.py`

""" 程序正确运行输出示例：
Dash is running on http://127.0.0.1:8080/

 * Serving Flask app 'server' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
"""
```

