from pyecharts import options as opts
from pyecharts.charts import Map3D
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode
from tkinter import *
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="3D地图", layout="wide")

#获取屏幕尺寸
base = Tk()
screen_height = base.winfo_screenheight()
screen_width = base.winfo_screenwidth()

example_data = [
    ("成都",[103.9526, 30.7617,5]),
    ("双流",[103.9237, 30.5744,5])
    ]

ditu = (
    Map3D(init_opts=opts.InitOpts(width=str(screen_width)+str("px"), height=str(screen_height)+str("px")))
    .add_schema(
        maptype="成都",
        itemstyle_opts=opts.ItemStyleOpts(
            color="rgb(5,91,123)",
            opacity=1,
            border_width=0.8,
            border_color="rgb(62,215,213)",
        ),
        map3d_label=opts.Map3DLabelOpts(
            is_show=True,
            text_style=opts.TextStyleOpts(
                 color="#fff", font_size=16, background_color="rgba(0,0,0,0)"
            ),
        ),
        emphasis_label_opts=opts.LabelOpts(
                is_show=False,
                color="#fff",
                font_size=10,
                background_color="rgba(0,23,11,0)",
            ),
 
        light_opts=opts.Map3DLightOpts(
            main_color="#fff",
            main_intensity=1.2,
            is_main_shadow=False,
            main_shadow_quality="high",
            main_beta=10,
            ambient_intensity=0.3,
        ),

        # view_control_opts=opts.Map3DViewControlOpts(center=[-10, 0, 10]),
        post_effect_opts=opts.Map3DPostEffectOpts(is_enable=False),
    )
    .add(
        series_name="",
        data_pair=example_data,
        type_=ChartType.BAR3D,
        bar_size=1,
        label_opts=opts.LabelOpts(
                is_show=True,
                formatter=JsCode(
                    "function(data){return data.name + ' ' + data.value[2];}"
                ),
                color="#fff",
                font_size=10,
            ),
        
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="3D地图"),
            visualmap_opts=opts.VisualMapOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(is_show=True),)
    .render_embed()
)

c1, c2, c3 = st.columns([0.6,5,1])
with c1:
    st.empty()
with c2:
    components.html(ditu, width=screen_width, height=screen_height)
with c3:
    st.empty()