# AI Stock Monitoring

一个面向个人投资者的轻量化 A 股监控系统，当前版本已经具备：

- `FastAPI` Web 服务与极简登录页
- `SQLite` 内置数据库
- 股票代码批量管理、邮箱配置、监控状态页面
- `AKShare` 实时行情、历史日线/周线、交易日历接入
- 250 日线、BOLL 中轨、30 周/60 周均线、近 12 个月股息率计算
- 开盘前股息率预警、盘中连续 2 次触发预警、周线交叉次日开盘提醒
- 告警历史、Web 弹窗确认、股票详情图表
- 交易流水录入、持仓重建、交易复盘与大模型分析
- 交易记录导出 Excel、复盘结果邮件推送

## 当前范围

这个版本已经能作为首个可用版本运行，但仍有一些后续优化项：

- 图表为极简 `canvas` 绘制，未引入专业前端图表库
- 邮件提醒依赖你手动填写的发件邮箱、收件邮箱与 SMTP 配置
- 交易复盘在配置 `OPENAI_API_KEY` 后会优先调用 OpenAI Responses API，否则自动回退到规则分析
- 暂未接入券商回单或自动同步成交记录，交易流水需要手动录入

## 本地启动

1. 创建并激活虚拟环境
2. 安装依赖：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
3. 如需大模型分析，可先执行：`cp .env.example .env` 并填写 `OPENAI_API_KEY`
4. 启动服务：`./start.sh` 或 `python main.py`
5. 打开：`http://127.0.0.1:1217`

默认管理员账号：

- 用户名：`admin`
- 密码：`admin123`

## Docker 启动

1. 复制模板：`cp .env.example .env`
2. 按需修改 `.env`（尤其是 `OPENAI_API_KEY`）
3. 构建并启动：`docker compose up -d --build`（Dockerfile 已默认优先使用清华 PyPI 镜像）
4. 打开：`http://127.0.0.1:1217`
5. 数据文件会落在宿主机 `./data/`

## 下载镜像源

- Docker 构建安装 Python 依赖时，默认优先使用清华 PyPI 镜像
- 本地安装依赖时，README 中也默认使用清华 PyPI 镜像命令
- 如果你后续想换成别的国内镜像，可以在 Docker 构建时覆盖 `PIP_INDEX_URL` 和 `PIP_TRUSTED_HOST`

## 环境模板

- 模板文件：`.env.example`
- 本地或 Docker 使用前建议先执行：`cp .env.example .env`
- `.env` 已加入 `.gitignore`，不会被误提交

## 关键环境变量

- `ASM_ADMIN_USERNAME`
- `ASM_ADMIN_PASSWORD`
- `ASM_HOST`
- `ASM_PORT`
- `ASM_DB_PATH`
- `ASM_REFRESH_INTERVAL`
- `ASM_PROVIDER`
- `ASM_DETAIL_CHART_DAYS`
- `ASM_LLM_PROVIDER`
- `ASM_LLM_MODEL`
- `ASM_LLM_BASE_URL`
- `OPENAI_API_KEY`

默认数据源为 `akshare`；如果想离线演示，可设置：`ASM_PROVIDER=mock`

## 交易复盘说明

交易复盘页支持手动录入：

- 股票代码
- 买入 / 卖出方向
- 成交价格
- 成交数量
- 成交时间
- 备注

系统会自动：

- 重建当前持仓数量、持仓均价、已实现盈亏
- 结合当前均线、股息率、触发状态给出复盘结果
- 在配置 `OPENAI_API_KEY` 后调用大模型，判断当前操作是否合理
- 给出下一步更稳健的买点、卖点和风控建议
- 一键导出交易记录 Excel，或把最新复盘结果推送到配置邮箱

## 测试

使用标准库运行：

```bash
python -m unittest discover -s tests -q
```
