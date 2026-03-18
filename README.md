# AI Stock Monitoring

一个面向个人投资者的轻量化 A 股监控系统，当前版本已经具备：

- `FastAPI` Web 服务与极简登录页
- `SQLite` 内置数据库
- 多账号登录、账号隔离的股票池 / 交易流水 / 邮箱配置 / 提醒历史 / 最近登录记录
- 股票代码批量管理、邮箱配置、监控状态页面
- `AKShare` 实时行情、历史日线/周线、交易日历接入
- 250 日线、BOLL 上/中/下轨、30 周/60 周均线、近 12 个月股息率计算
- 开盘前股息率预警、盘中连续 2 次触发预警、周线交叉次日开盘提醒
- 新增卖出侧提醒：突破 BOLL 上轨、股息率低于 3.5%、量化走弱卖出
- 每只股票、每个指标每天只发送一次提醒，避免同日重复轰炸邮箱
- 告警历史、Web 弹窗确认、股票详情图表
- 交易流水录入、持仓重建、交易复盘与大模型分析
- 可选量化盈利概率提醒：支持阈值、动量、波动率、股息率、BOLL 偏离、多模型组合打分，并可触发量化走弱卖出提醒
- 新增轻量专业基准模型：内置 MSCI 风格动量、质量稳定因子，并与自适应模型做滚动回测对照
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
5. 打开：`http://127.0.0.1:11223`

默认管理员账号：

- 用户名：`admin`
- 密码：`admin123`

说明：

- `ASM_ADMIN_USERNAME` / `ASM_ADMIN_PASSWORD` 只在数据库首次初始化时作为种子值写入
- 后续请在 Web 页面里修改管理员密码，修改结果会持久化，不会在重启后被默认值覆盖

## Docker 启动

1. 如需保留一份本地环境文件，可先执行：`./prepare_env.sh`（会在缺失时自动生成 `.env`）
2. 按需修改 `.env`（尤其是 `OPENAI_API_KEY`、`ASM_PUBLIC_DOMAIN`、`ASM_TLS_EMAIL`）
3. 直接构建并启动应用：`docker compose up -d --build`（默认只启动应用本体，不再因为 HTTPS 代理镜像拉取失败而整体报错）
4. 如需 HTTPS，请填写 `ASM_PUBLIC_DOMAIN` 和 `ASM_TLS_EMAIL`，并确保域名已解析到服务器、80/443 端口已放行
5. 建议先执行一次：`sudo ./docker-enable-mirrors.sh`，为 Docker Daemon 配置国内镜像加速器
6. 启动 HTTPS 版本：`./start-https.sh` 或 `COMPOSE_PROFILES=https docker compose up -d --build`
7. 本地直连应用可打开：`http://127.0.0.1:11223`
8. 外网 HTTPS 应访问：`https://你的域名`，不要再访问 `https://你的域名:11223`
9. 数据文件和 Caddy 证书会落在宿主机 `./data/`
10. 日常升级可执行：`./upgrade.sh`（会先自动生成 `.env`，再备份数据库并重建容器）

## 下载镜像源

- Docker 基础镜像默认改为国内镜像：`swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.12-slim`
- Python 基础镜像仍默认使用国内源：`swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.12-slim`
- `Caddy` 默认改回官方镜像 `caddy:2`，并建议通过 Docker Daemon 镜像加速器来解决拉取问题
- 如果你的服务器对某个镜像源访问不稳定，可以在 `.env` 中自行改 `BASE_IMAGE` 和 `CADDY_IMAGE`
- Docker 构建安装 Python 依赖时，默认优先使用清华 PyPI 镜像
- 本地安装依赖时，README 中也默认使用清华 PyPI 镜像命令
- 如果你后续想换成别的国内镜像，可以在 Docker 构建时覆盖 `PIP_INDEX_URL` 和 `PIP_TRUSTED_HOST`

## HTTPS 说明

- 当前 Docker 方案内置可选的 `Caddy` 反向代理，自动申请和续期 HTTPS 证书
- 应用容器仍然运行在 `11223` HTTP 端口；只有启用 `https` profile 后，`Caddy` 才会对外提供标准 `443` HTTPS
- 如果你用域名部署，正确访问方式应是：`https://你的域名`
- 登录自助解封验证码只会发送到当前账号在 Web 页面填写的“收件邮箱”，不会读取任何写死的固定邮箱
- `https://你的域名:11223` 仍然会失败，因为 `11223` 端口上跑的是纯 HTTP，不是 TLS
- 首次签发证书前，请确认域名 A/AAAA 记录已经指向服务器公网 IP，且云防火墙/安全组已开放 `80` 和 `443`

## 环境模板

- 模板文件：`.env.example`
- 本地或 Docker 使用前建议先执行：`cp .env.example .env`
- `.env` 已加入 `.gitignore`，不会被误提交
- `./prepare_env.sh` 会在缺少 `.env` 时自动由 `.env.example` 生成

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

## 多账号与量化提醒

- 管理员可在 Web 页面直接新增账号
- 每个账号的数据默认隔离：监控股票、快照、提醒历史、交易流水、邮箱配置、量化阈值互不共享
- 量化提醒页支持开启/关闭、设置提醒阈值，并勾选多个组合模型
- 可进一步调节价格是否必须站上 250 日线、周线多头过滤、最低股息率、20 日动量、20 日波动率、BOLL 偏离阈值
- 当前内置模型包括：趋势跟随、均值回归、股息质量、周线共振、波动过滤
- 当前还支持专业基准模型：MSCI 动量、质量稳定，并能和自适应模型按最近历史表现动态调权
- “盈利概率”是模型综合评分，不是对未来收益的真实承诺

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
