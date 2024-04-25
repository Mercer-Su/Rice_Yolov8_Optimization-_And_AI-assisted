from ast import Return
from pyexpat.errors import messages
from ultralytics import YOLO
from http import HTTPStatus
import cv2
from PIL import Image
import numpy as np
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import json
import streamlit as st
import pandas as pd

import dashscope

dashscope.api_key = "sk-6ca9b64b3b004a6c9fad176e7ba2b446"

@st.cache_resource
def load_map():
    path = "./city_map.json"
    with open(path, 'r',encoding='utf-8') as f:
        map_data = json.load(f)
    
    seq = map_data['seq']
    tree = map_data['tree']
    return tree,seq

def load_static():
    path = "./static.json"
    with open(path, 'r',encoding='utf-8') as f:
        static_data = json.load(f)
    return static_data

def save_static(static_data):
    path = "./static.json"
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(static_data,f,ensure_ascii=False,indent=4)
    f.close()

def make_map(choose,static_map):
    city,block = choose
    city_key = str(city)
    block_key = str(block)

    city_info = static_map[city_key]
    block_info = static_map[block_key]

    import numpy as np

    data = [{"lat": city_info["loc"][1],"lon":city_info["loc"][0],"size":100,"color":np.random.rand(4).tolist()},
            {"lat":block_info["loc"][1],"lon":block_info["loc"][0],"size":500,"color":np.random.rand(4).tolist()},
            {"lat":block_info["loc"][1]+0.01,"lon":block_info["loc"][0]+0.01,"size":800,"color":np.random.rand(4).tolist()}
            ]

    return pd.DataFrame(data)



def infer_uploaded_image(conf, model,source_img=None,ban_btn=False):
    """
    执行图片推理
    :param conf: Confidence of YOLO model
    :param model: An instance of the `YOLO class containing the YOLO model.
    :return: None
    """
    if source_img is None:
        source_img = st.file_uploader(
            label="选择一张图片...",
            type=("jpg", "jpeg", "png", 'bmp', 'webp')
        )

    #页面划分一个区域用于显示检测前后的图片（一行两列）
    col1, col2 = st.columns(2)
    with col1:
        if source_img is not None:
            if isinstance(source_img, np.ndarray):
                uploaded_image = Image.fromarray(source_img)
            else:
                uploaded_image = Image.open(source_img)
            # 将原始图片显示在页面中(col1位置)
            st.image(
                image=source_img,
                caption="上传图片",
                use_column_width=True
            )

    if source_img is not None:
        
        if not ban_btn:
            st.button("执行",use_container_width=True)

        with st.spinner("执行中..."):
            """yolo预测返回的结果全在返回变量res中
            这是一个ultralytics.yolo.engine.results.Results 类
            该类包含了检测、分割、关键点检测、分类任务的所有预测信息
            Results类中names属性:类别标签
            boxes属性:目标检测信息
            masks：实例分割信息
            probs：图像分类信息
            keypoints：人体关键点检测信息
            具体信息可以参看 ultralytics.yolo.engine.results 路径中的Results类
            """
            res = model.predict(uploaded_image,
                                conf=conf)
            print(res)
            labels=res[0].names
            print(labels)
            boxes = res[0].boxes
            #关于下面的plot()见  本类中 _display_detected_frames方法对它解释(在下面)
            res_plotted = res[0].plot()[:, :, ::-1]
            print(type(res_plotted))
            with col2:
                st.image(res_plotted,
                        caption="Detected Image",
                        use_column_width=True)
                try:
                    #统计一张图片中Label 个数及数量 显示在前端页面中
                    with st.expander("检测结果",expanded=False):
                        labels_num_dict={}
                        for box in boxes:
                            print(box)
                            lable_index=box.cls.cpu().detach().numpy()[0].astype(int)
                            for key in labels.keys():
                                if int(lable_index)==key:
                                    if labels[key] in labels_num_dict:
                                        labels_num_dict[labels[key]]+=1
                                    else:
                                        labels_num_dict[labels[key]] = 1
                        return labels_num_dict
                except Exception as ex:
                    st.write("没有选择待检测的图片!")
                    st.write(ex)
                    

def _display_detected_frames(conf, model, st_frame,st_text, image):
    """
    逐帧推理
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # 设置每一帧(图片)合适的大小  显示在页面中
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # yolo模型推理预测
    res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    """
    关于这个plot()：它是ultralytics.yolo.engine.results.Results类中定义的方法，下面是这个方法的官方注释
    Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.
        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            img_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability
        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
    """
    # 将后处理的检测目标(带框)逐帧显示在页面上
    st_frame.image(res_plotted,
       caption='Detected Video',
       channels="BGR",
       use_column_width=True
       )
    #这里也可以把yolo预测的所有信息、所有数据全部写到页面中
    # try:
    #    with st.expander("检测结果", expanded=False):
    #        st_text.write(res[0])
    # except Exception as e:
    #        st_text.error(e)
@st.cache_resource
def load_model(model_path):
    """
    从具体的路径中加载一个模型
    Parameters:
        model_path (str): The path to the YOLO model file.
    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def infer_uploaded_video(conf, model):
    """
    执行视频推理
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.file_uploader(
        label="选择一个视频..."
    )
    #在页面中显示传入的原始视频
    if source_video:
        st.video(source_video)


    if source_video:
        # try:
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(source_video.read())
        vid_cap = cv2.VideoCapture(
            tfile.name)

        # 页面创建两个空容器一个实时播放画面一个实时展示信息
        # st_frame = st.empty()
        # st_text=st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                # 调用方法逐帧预测
                # _display_detected_frames(conf,
                #                          model,
                #                          st_frame,
                #                          st_text,
                #                          image
                #                          )
                predict = infer_uploaded_image(conf, model, image[:,:,::-1],True)
                if predict:
                    return predict
            else:
                vid_cap.release()
                break
        # except Exception as e:
        #     st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    执行实时摄像头推理
    :param conf: Confidence of YOLO model
    :param model: An instance of the `YOLO` class containing the YOLO model.
    :return: None
    """
    img = st.camera_input("拍照")
    if img:
        predict = infer_uploaded_image(conf, model, img)
        return predict
    
    # try:
    #     flag = st.button(
    #         label="终止执行"
    #     )
    #     vid_cap = cv2.VideoCapture(0)  #调用本地摄像头
    #     #页面创建两个空容器一个实时播放画面一个实时展示信息
    #     st_frame = st.empty()
    #     st_text=st.empty()
    #     while not flag:
    #         success, image = vid_cap.read()
    #         if success:
    #             #调用方法逐帧预测
    #             _display_detected_frames(
    #                 conf,
    #                 model,
    #                 st_frame,
    #                 st_text,
    #                 image
    #             )
    #         else:
    #             vid_cap.release()
    #             break
    # except Exception as e:
    #     st.error(f"Error loading video: {str(e)}")

#         def transform(self, frame):
#             img = frame.to_ndarray(format="bgr24")  # 将帧转换为ndarray格式
#
#             # 这里应该是你的模型推理代码
#             # 例如: results = model.predict(img)
#             # 然后根据置信度过滤结果
#             # filtered_results = filter(lambda x: x['confidence'] >= conf, results)
#             # 注意：你需要自己实现模型推理和结果过滤的逻辑
#             # 这里只是一个示例
#
#             # 如果需要在视频帧上绘制结果，可以在这里添加绘制代码
#             # 例如: img = draw_results(img, filtered_results)
#
#             # 将处理后的帧返回到视频流中
#             return av.VideoFrame.from_ndarray(img, format="bgr24")

class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 使用YOLO模型进行推理
        res = self.model.predict(img, conf=self.conf)
        res_plotted = res[0].plot()  # 假设plot返回的是BGR图像

        # 将处理后的帧返回到视频流中
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")



def call_with_messages(text,pred):

    template = '''
    你是一名灾害防治专家，请你根据天气情况和灾害情况进行建议

    当前天气情况：{text}

    当前灾害情况：{pred}

    '''

    messages = [{"role":"user","content":template.format(text=text,pred=pred)}]

    response = dashscope.Generation.call(
        # dashscope.Generation.Models.qwen_turbo,
        "qwen1.5-1.8b-chat",
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        temperature=0.95,
        top_p=0.8,
        max_tokens=200,
    )
    if response.status_code == HTTPStatus.OK:
        res = response["output"]["choices"][0]["message"]["content"]
    else:
        res = '请求异常'
    return res


# 使用WebRTC进行网络摄像头视频流处理
# def infer_uploaded_webcam_http(conf, model):
#     # WebRTC配置
#     rtc_configuration = RTCConfiguration(
#         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#     )
#
#     # 创建WebRTC流
#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         video_processor_factory=VideoProcessor,
#         rtc_configuration={  # Add this line
#             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         }
#     if webrtc_ctx.video_transformer:
#         st.text("摄像头正在运行...")
#         if st.button("终止执行"):
#             webrtc_ctx.state.playing = False  # 停止视频流
#     else:
#         st.text("点击下方按钮开始摄像头")
#         if st.button("开始执行"):
#             webrtc_ctx.state.playing = True  # 开始视频流
