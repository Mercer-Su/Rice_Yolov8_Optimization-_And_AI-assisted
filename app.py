from pathlib import Path


import config
from utils import infer_uploaded_video, load_model, infer_uploaded_image,infer_uploaded_video,infer_uploaded_webcam,load_map,load_static,make_map,call_with_messages,save_static

import random
import datetime
import base64

import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyecharts.globals import ChartType

import graphviz
import pydeck as pdk
import altair as alt
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import streamlit_antd_components as sac

from pyecharts.charts import *
from pyecharts.globals import ThemeType
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from streamlit_echarts import st_pyecharts

import warnings

import streamlit as st
from streamlit_webrtc import webrtc_streamer


# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="st.cache is deprecated", category=DeprecationWarning)


def main():
    # 设置页面布局
    global model
    st.set_page_config(
        page_title="Coding Learning Corner",
        page_icon="log/头像专用.jpg",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 设置主标题
    st.title("水稻种植辅助系统")

    # 侧边栏标题
    st.sidebar.header("模型配置")

    # 侧边栏-任务选择
    task_type = st.sidebar.selectbox(
        "选择要进行的任务",
        ["目标检测"]
    )

    model_type = None
    # 侧边栏-模型选择
    if task_type == "目标检测":
        model_type = st.sidebar.selectbox(
            "选取模型",
            config.DETECTION_MODEL_LIST
        )
    else:
        st.error("目前仅仅实现了目标检测任务")
    # 侧边栏-置信度
    confidence = float(st.sidebar.slider(
        "选取最小置信度", 10, 100, 25)) / 100

    model_path = ""
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    else:
        st.error("请在下拉框选择一个模型")

    # 加载模型
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"无法加载模型. 请检查路径: {model_path}")

    # 侧边栏-图像、视频、摄像头选择
    predict_region, show_region = st.columns(2)

    with predict_region:
        # st.header("图片/视频配置")
        source_selectbox = st.selectbox(
            "选取文件类型",
            config.SOURCES_LIST
        )

        source_img = None
        predict = None
        if source_selectbox == config.SOURCES_LIST[1]:  # 摄像头
            predict = infer_uploaded_webcam(confidence, model)
        elif source_selectbox == config.SOURCES_LIST[0]:  # 图片
            predict = infer_uploaded_image(confidence, model)
        elif source_selectbox == config.SOURCES_LIST[2]:  # 
            predict = infer_uploaded_video(confidence, model)
        else:
            st.error("目前仅支持 '图片' '视频' '本地摄像头' ")

    charts_mapping = {
        '线性表': 'line_chart', '条纹图': 'bar_chart', '地区': 'area_chart', '直方图': 'pyplot',
        'Altair': 'altair_chart',
        '地图': 'map', '密度图': 'plotly_chart', '柱状密度': 'pydeck_chart', 'Graphviz': 'graphviz_chart',
        '比例图': '',
        'PyEchart': ''
    }
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
    else:
        st.session_state.first_visit = False
    # 初始化全局配置
    if st.session_state.first_visit:
        # 在这里可以定义任意多个全局变量，方便程序进行调用
        st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(
            hours=0)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
        st.session_state.random_chart_index = random.choice(range(len(charts_mapping)))
        st.session_state.my_random = MyRandom(random.randint(1, 1000000))
        st.session_state.city_mapping,_ = get_city_mapping()
        st.session_state.static = load_static()
        st.session_state.city_map, st.session_state.city_seq = load_map()
        st.balloons()
        st.snow()

    d = st.sidebar.date_input('Date', st.session_state.date_time.date())
    t = st.sidebar.time_input('Time', st.session_state.date_time.time())
    t = f'{t}'.split('.')[0]
    st.sidebar.write(f'The current date time is {d} {t}')
    # with predict_region:
    #     chart = st.selectbox('选择你想查看的图表', charts_mapping.keys(),
    #                              index=st.session_state.random_chart_index)
    with show_region:
        city_choose = sac.cascader(st.session_state.city_map,index=[2090,2106],return_index=True,search=True)
        city_index = city_choose[0]
        city = st.session_state.city_seq[city_index]

        cat = st.selectbox("灾害类别:",config.categories_map,format_func=config.categories_map.get)

        for p in st.session_state.city_map:
            if p['label'] == st.session_state.city_seq[city_index]:
                children_size = len(p['children'])
                break

        map_static_data = make_map_static(cat,city_index,children_size,st.session_state.city_seq,st.session_state.static)
    
        c = map3d_with_bar3d(map_static_data,tag=2)

        # components.html(c,height=500,width=700)
    

    # color = st.sidebar.color_picker('选择你的偏好颜色', '#520520')
    # st.sidebar.write('当前的颜色是', color)

    with st.container():
        st.markdown(f'### {city} 天气预测')
        forecastToday, df_forecastHours, df_forecastDays = get_city_weather(st.session_state.city_mapping[city])
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric('天气情况', forecastToday['weather'])
        col2.metric('当前温度', forecastToday['temp'])
        col3.metric('当前体感温度', forecastToday['realFeel'])
        col4.metric('湿度', forecastToday['humidity'])
        col5.metric('风向', forecastToday['wind'])
        col6.metric('预测更新时间', forecastToday['updateTime'])
        st.markdown(
            """
            <style>
            body{
                if(weather=="多云"):
                    background-image:url('https://i.imgur.com')
            }
            </style>
            """

        )


        c1 = (
            Line()
            .add_xaxis(xaxis_data=df_forecastHours.index.to_list())
            .add_yaxis(series_name='Temperature', y_axis=df_forecastHours['Temperature'].values.tolist())
            .add_yaxis(series_name='Body Temperature', y_axis=df_forecastHours['Body Temperature'].values.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title="24小时预测情况"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value", axislabel_opts=opts.LabelOpts(formatter="{value} °C")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(is_show=True, formatter=JsCode("function(x){return x.data[1] + '°C';}")))
        )

        c2 = (
            Line()
            .add_xaxis(xaxis_data=df_forecastDays.index.to_list())
            .add_yaxis(series_name="High Temperature", y_axis=df_forecastDays.Temperature.apply(
                lambda x: int(x.replace('°C', '').split('~')[1])).values.tolist())
            .add_yaxis(series_name="Low Temperature", y_axis=df_forecastDays.Temperature.apply(
                lambda x: int(x.replace('°C', '').split('~')[0])).values.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title="7 Days Forecast"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="value", axislabel_opts=opts.LabelOpts(formatter="{value} °C")),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(is_show=True, formatter=JsCode("function(x){return x.data[1] + '°C';}")))
        )

        t = Timeline(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='1200px'))
        t.add_schema(play_interval=10000, is_auto_play=True)
        t.add(c1, "24小时预报")
        t.add(c2, "7天预报")
        components.html(t.render_embed(), width=1200, height=520)
        with st.expander("24小时预报"):
            st.table(
                df_forecastHours.style.format({'Temperature': '{}°C', 'Body Temperature': '{}°C', 'Humidity': '{}%'}))
        with st.expander("7天预报", expanded=True):
            st.table(df_forecastDays)

    with predict_region:
        if predict:
            predict_items = []
            key_index = str(city_choose[-1])
            for p in predict:
                p_metric = config.categories_map[p]
                predict_items.append(p_metric)
                if p not in st.session_state.static[key_index]["static"]:
                    st.session_state.static[key_index]["static"][p] = 1
                else:
                    st.session_state.static[key_index]["static"][p] += 1
            predict_desc = ",".join(predict_items)
            ai_call_message = call_with_messages(str(forecastToday),predict_desc)
            st.text_area("AI决策防治:",ai_call_message, height=350)

            save_static(st.session_state.static)
            
    if predict:
        with show_region:
            make_categories_pie(city_choose,st.session_state.static)
    else:
        with predict_region:
            make_categories_pie(city_choose,st.session_state.static)

    # st.markdown(f'### {chart} 图表')
    # df = get_chart_data(chart, st.session_state.my_random)
    # eval(
    #     f'st.{charts_mapping[chart]}(df{",use_container_width=True" if chart in ["Distplot", "Altair"] else ""})' if chart != 'PyEchart' else f'st_echarts(options=df)')


class MyRandom:
    def __init__(self, num):
        self.random_num = num


def my_hash_func(my_random):
    num = my_random.random_num
    return num


# @st.cache(hash_funcs={MyRandom: my_hash_func}, allow_output_mutation=True, ttl=3600,suppress_st_warning=True)
# def get_chart_data(chart, my_random):
#     data = np.random.randn(20, 3)
#     df = pd.DataFrame(data, columns=['a', 'b', 'c'])
#     if chart in ['线性表', '条纹图', '地区']:
#         return df

#     elif chart == '直方图':
#         arr = np.random.normal(1, 1, size=100)
#         fig, ax = plt.subplots()
#         ax.hist(arr, bins=20)
#         return fig

#     elif chart == 'Altair':
#         df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
#         c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
#         return c

#     elif chart == '地图':
#         df = pd.DataFrame(np.random.randn(1000, 2) / [5, 5] + [30.66, 104.07], columns=['lat', 'lon'])
#         return df

#     elif chart == '密度图':
#         x1 = np.random.randn(200) - 2
#         x2 = np.random.randn(200)
#         x3 = np.random.randn(200) + 2
#         # Group data together
#         hist_data = [x1, x2, x3]
#         group_labels = ['Group 1', 'Group 2', 'Group 3']
#         # Create distplot with custom bin_size
#         fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
#         # Plot!
#         return fig

#     elif chart == '柱状密度':
#         df = pd.DataFrame(np.random.randn(1000, 2) / [25, 25] + [30.66, 104.07], columns=['lat', 'lon'])
#         args = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
#                         initial_view_state=pdk.ViewState(latitude=30.66, longitude=104.07, zoom=11, pitch=50, ),
#                         layers=[
#                             pdk.Layer('HexagonLayer', data=df, get_position='[lon, lat]', radius=200, elevation_scale=4,
#                                       elevation_range=[0, 1000], pickable=True, extruded=True),
#                             pdk.Layer('ScatterplotLayer', data=df, get_position='[lon, lat]',
#                                       get_color='[200, 30, 0, 160]', get_radius=200)])
#         return args

#     elif chart == 'Graphviz':
#         graph = graphviz.Digraph()
#         graph.edge('grandfather', 'father')
#         graph.edge('grandmother', 'father')
#         graph.edge('maternal grandfather', 'mother')
#         graph.edge('maternal grandmother', 'mother')
#         graph.edge('father', 'brother')
#         graph.edge('mother', 'brother')
#         graph.edge('father', 'me')
#         graph.edge('mother', 'me')
#         graph.edge('brother', 'nephew')
#         graph.edge('Sister-in-law', 'nephew')
#         graph.edge('brother', 'niece')
#         graph.edge('Sister-in-law', 'niece')
#         graph.edge('me', 'son')
#         graph.edge('me', 'daughter')
#         graph.edge('where my wife?', 'son')
#         graph.edge('where my wife?', 'daughter')
#         return graph

#     elif chart == '比例图':
#         rain = [0.1, 4.6, 5.8, 14.2, 16.3, 25.3, 34.5, 45.2, 41.0, 16.3, 9.9, 4.1]
#         month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#         data_pari_rain = [list(data) for data in zip(month, rain)]  # month 相当于自变量, rain 相当于因变量
#         pie = Pie()
#         pie.add(
#             series_name="平均降水",
#             data_pair=data_pari_rain,
#             radius="50%",
#             center=["50%", "50%"],
#             label_opts=opts.LabelOpts(is_show=False, position="center")
#         )
#         pie.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
#         pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
#         pie.render("Pie_basic.html")
#         with open("Pie_basic.html", 'r') as p:
#             a = p.read()
#             components.html(a, height=1000, width=1000)
#         p.close()



#     elif chart == 'PyEchart':
#         options = {
#             "xAxis": {
#                 "type": "category",
#                 "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
#             },
#             "yAxis": {"type": "value"},
#             "series": [
#                 {"data": [820, 932, 901, 934, 1290, 1330, 1320], "type": "line"}
#             ],
#         }
#         return options


@st.cache_data(ttl=3600)
def get_city_mapping():
    url = 'https://h5ctywhr.api.moji.com/weatherthird/cityList'
    r = requests.get(url)
    data = r.json()
    city_mapping = dict()
    chengdu = 0
    flag = True
    for i in data.values():
        for each in i:
            city_mapping[each['name']] = each['cityId']
            if each['name'] != '成都市' and flag:
                chengdu += 1
            else:
                flag = False

    return city_mapping, chengdu


@st.cache_data(ttl=3600)
def get_city_weather(cityId):
    url = 'https://h5ctywhr.api.moji.com/weatherDetail'
    headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
    data = {"cityId": cityId, "cityType": 0}
    r = requests.post(url, headers=headers, json=data)
    result = r.json()

    # today forecast
    forecastToday = dict(
        humidity=f"{result['condition']['humidity']}%",
        temp=f"{result['condition']['temp']}°C",
        realFeel=f"{result['condition']['realFeel']}°C",
        weather=result['condition']['weather'],
        wind=f"{result['condition']['windDir']}{result['condition']['windLevel']}级",
        updateTime=(datetime.datetime.fromtimestamp(result['condition']['updateTime']) + datetime.timedelta(
            hours=8)).strftime('%H:%M:%S')
    )

    # 24 hours forecast
    forecastHours = []
    for i in result['forecastHours']['forecastHour']:
        tmp = {}
        tmp['PredictTime'] = (datetime.datetime.fromtimestamp(i['predictTime']) + datetime.timedelta(hours=8)).strftime(
            '%H:%M')
        tmp['Temperature'] = i['temp']
        tmp['Body Temperature'] = i['realFeel']
        tmp['Humidity'] = i['humidity']
        tmp['Weather'] = i['weather']
        tmp['Wind'] = f"{i['windDesc']}{i['windLevel']}级"
        forecastHours.append(tmp)
    df_forecastHours = pd.DataFrame(forecastHours).set_index('PredictTime')

    # 7 days forecast
    forecastDays = []
    day_format = {1: '昨天', 0: '今天', -1: '明天', -2: '后天'}
    for i in result['forecastDays']['forecastDay']:
        tmp = {}
        now = datetime.datetime.fromtimestamp(i['predictDate']) + datetime.timedelta(hours=8)
        diff = (st.session_state.date_time - now).days
        festival = i['festival']
        tmp['PredictDate'] = (day_format[diff] if diff in day_format else now.strftime('%m/%d')) + (
            f' {festival}' if festival != '' else '')
        tmp['Temperature'] = f"{i['tempLow']}~{i['tempHigh']}°C"
        tmp['Humidity'] = f"{i['humidity']}%"
        tmp['WeatherDay'] = i['weatherDay']
        tmp['WeatherNight'] = i['weatherNight']
        tmp['WindDay'] = f"{i['windDirDay']}{i['windLevelDay']}级"
        tmp['WindNight'] = f"{i['windDirNight']}{i['windLevelNight']}级"
        forecastDays.append(tmp)
    df_forecastDays = pd.DataFrame(forecastDays).set_index('PredictDate')
    return forecastToday, df_forecastHours, df_forecastDays


def get_v2_text():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    r = requests.get('https://bulinkbulink.com/freefq/free/master/v2', headers=headers, verify=False)
    v2_text = base64.b64decode(r.text).decode('utf-8')
    return v2_text


def make_categories_pie(choose,static_map):
    _,block = choose
    block_key = str(block)
    block_info = static_map[block_key]

    data = []

    for c,v in config.categories_map.items():
        data.append([v, block_info["static"].get(c,0)])

    pie = (
        Pie()
        .add("", data)
        .set_global_opts(title_opts=opts.TitleOpts(is_show=False))
        .set_colors(["#008B8B","#97FFFF","#FFE4E1","#4169E1","#66CD00","#B3EE3A","#BC8F8F","#FF6A6A","#FF8247","#FF4500"])
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )

    st_pyecharts(pie,height="500px")

def map3d_with_bar3d(example_data,tag=1) -> Map3D:
    if tag==1:
        ditu = (
            Map3D()
            .add_schema(
                maptype="成都",
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#b0a0f4",
                    opacity=1,
                    border_width=0.8,
                    border_color="rgb(62,215,213)",
                ),
                map3d_label=opts.Map3DLabelOpts(
                    is_show=False,
                    text_style=opts.TextStyleOpts(
                        color="#fff", font_size=16, background_color="rgba(0,0,0,0)"
                    ),
                ),
                # emphasis_label_opts=opts.LabelOpts(
                #     is_show=False,
                #     color="white",
                #     font_size=25,
                #     ),
        
                light_opts=opts.Map3DLightOpts(
                    main_color="white",
                    main_intensity=1.0,
                    is_main_shadow=False,
                    main_shadow_quality="high",
                    main_beta=10,
                    ambient_intensity=0.3,
                ),

                # view_control_opts=opts.Map3DViewControlOpts(center=[0, 0, 1]),
                # post_effect_opts=opts.Map3DPostEffectOpts(is_enable=False),
            )
            .add(
                series_name="",
                data_pair=example_data,
                type_=ChartType.BAR3D,
                # bar_size=0.5,
                shading="lambert",
                label_opts=opts.LabelOpts(
                        is_show=True,
                        formatter=JsCode(
                            "function(data){return data.name + data.value[2];}"
                        ),
                        color="white",
                        font_size=15,
                    ),
                
            )
            .set_global_opts(title_opts=opts.TitleOpts(title="分布情况"),
                    visualmap_opts=opts.VisualMapOpts(is_show=False),
                    tooltip_opts=opts.TooltipOpts(is_show=True),
                    )
            .render_embed()
        )
        components.html(ditu,height=500,width=700)
    if tag==2:
        base_data = []
        scaling = 50
        for d in example_data:
            base_data.extend([[d[0],*d[1]] for i in range(max(d[1][-1]//scaling,1))])

        df = pd.DataFrame(base_data,columns = ["name","lon","lat","value"])
        colors = [np.random.rand(4).tolist()]*len(df)
        colors[0] = np.random.rand(4).tolist()
        df["color"] = colors

        df_size = len(df)

        df["lat"] = df["lat"] + np.random.randn(df_size)/50
        df["lon"] = df["lon"] + np.random.randn(df_size)/50

        st.map(df,longitude="lon",latitude="lat",size="value",color="color",zoom=10)


def make_map_static(cat,city_index,children_size,city_seq,static):
    city_key = str(city_index)
    data = [(city_seq[city_index],[*static[city_key]["loc"],10])]
    for p in range(city_index+1,city_index+children_size):
        p_key = str(p)
        p_info = static[p_key]
        if cat in p_info["static"]:
            data.append((city_seq[p],[*p_info["loc"],p_info["static"][cat]]))
    return data

if __name__ == '__main__':
    main()
