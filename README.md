# Streamlit News Judgement Demo

这是一个简单的Streamlit应用程序，用于文本输入和新闻真实性判断。

## 项目结构

```
streamlit-project
├── src
│   └── streamlit.py      # Streamlit应用的主要代码
├── Dockerfile             # Docker镜像构建文件
├── requirements.txt       # 项目所需的Python依赖包
└── README.md              # 项目的文档和使用说明
```

## 安装依赖

在项目根目录下，使用以下命令安装所需的依赖包：

```
pip install -r requirements.txt
```

## 运行应用

要运行Streamlit应用，请使用以下命令：

```
streamlit run src/streamlit.py
```

## Docker支持

该项目包含一个Dockerfile，可以通过Docker构建和运行应用。使用以下命令构建Docker镜像：

```
docker build -t streamlit-news-judgement .
```

然后使用以下命令运行Docker容器：

```
docker run -p 8501:8501 streamlit-news-judgement
```

在浏览器中访问 `http://localhost:8501` 以查看应用。

## 贡献

欢迎任何形式的贡献！请提交问题或拉取请求。

## 许可证

此项目采用MIT许可证。