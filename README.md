# Cheatsheet AI Assistant

Cheatsheet AI Assistant is a Streamlit app that turns lecture slides, notes, PDFs, and documents into a compact, exam-oriented cheat sheet.

## 快速开始

推荐直接使用项目里已经补好的 `Makefile`：

```bash
make install
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
make run
```

启动后，终端会显示本地地址，通常是 `http://localhost:8501`。

## Python 版本说明

当前这个目录在本机是用 `Python 3.8.2` 运行的，所以我已经把 `Streamlit` 依赖约束调整为与 `Python 3.8` 兼容的版本范围。
如果你以后切到 `Python 3.9+`，也可以再把 `streamlit` 版本升级到更新的分支。
现在 `requirements.txt` 已经按 Python 版本做了兼容处理：本地 `3.8` 会安装兼容版本，云端 `3.9+` 会安装较新的 `Streamlit`。

## 手动搭建 Streamlit

如果你不想用 `make`，也可以手动执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI 配置

这个项目现在同时支持两种配置方式：

1. 环境变量

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4.1-mini"
```

2. Streamlit secrets

先复制示例文件：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

然后把 `.streamlit/secrets.toml` 改成这样：

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-4.1-mini"
```

如果没有配置 OpenAI Key，应用也能运行，只是会自动切到本地启发式模式，生成质量会弱一些。

### Streamlit Cloud 里怎么填

你截图里的 `Advanced settings -> Secrets` 文本框，直接粘贴下面两行就可以：

```toml
OPENAI_API_KEY = "sk-你的真实key"
OPENAI_MODEL = "gpt-4.1-mini"
```

也支持这种分组写法：

```toml
[openai]
api_key = "sk-你的真实key"
model = "gpt-4.1-mini"
```

这两种格式项目都已经支持，而且 `secrets` 不会进入 Git 仓库。

## 项目结构

```text
app.py
cheatsheet_ai/
  __init__.py
  extractors.py
  processing.py
  generator.py
  exporters.py
.streamlit/
  config.toml
  secrets.toml.example
Makefile
requirements.txt
README.md
```

## 功能概览

- 支持上传 `PDF`、`PPTX`、`DOCX`、`TXT`
- 自动抽取和清洗课程资料文本
- 长文本会先分块再汇总
- 可按语言、篇幅、重点方向和细节密度生成小抄
- 支持在网页里直接编辑结果
- 支持导出 `Markdown`、`PDF`、`DOCX`

## Streamlit 项目内已补充的内容

- `.streamlit/config.toml`
  让项目具备默认主题、自动保存刷新和更标准的 Streamlit 配置
- `.streamlit/secrets.toml.example`
  方便本地和部署时填写密钥
- `Makefile`
  提供 `make install`、`make run`、`make check`
- `.gitignore`
  忽略 `.venv`、`secrets.toml` 和缓存文件

## 校验命令

可以用下面的命令做一次基础检查：

```bash
make check
```

## 运行模式

- `OpenAI mode`
  配置了 `OPENAI_API_KEY` 后，会调用 OpenAI 进行分块总结和最终生成
- `Heuristic mode`
  没有配置密钥时，仍然能跑通完整流程，但更适合演示或离线原型
