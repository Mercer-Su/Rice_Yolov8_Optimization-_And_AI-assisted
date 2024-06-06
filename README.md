## 使用环境：

 - Anaconda 3
 - Python 3.9
 - Pytorch 11.8
 - Windows 11

## 安装环境

 - 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**建议源码安装**，源码安装能保证使用最新代码。

```shell
git clone https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch.git
cd VoiceprintRecognition-Pytorch/
pip install .
```

**安装requirements.txt文件里的依赖**

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```
