import base64
import time

import cv2
from dash import html
from dash import dcc
import dash_uploader as du
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import config
import utils
import render
from server import app, server
from PIL import Image
import flask
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import os.path as osp

from config import *

du.configure_upload(app, folder='data')


def get_graph_figure(_img):
    fig = px.imshow(_img)
    fig.update_layout(
        autosize=True,
        # height=800,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        paper_bgcolor='#f9f9f8',
        plot_bgcolor='#f9f9f8',
        showlegend=False,
        xaxis={
            # 'range': [0.2, 1],
            'showgrid': False,  # thin lines in the background
            'zeroline': False,  # thick line at x=0
            'visible': False,  # numbers below
        },
        yaxis={
            # 'range': [0.2, 1],
            'showgrid': False,  # thin lines in the background
            'zeroline': False,  # thick line at x=0
            'visible': False,  # numbers below
        },
    )
    return fig


@du.callback(
    Output("div-graph-picture", "children"),
    id='uploader',
)
def update_graph_picture(filenames):
    img = Image.open(os.path.join(PATH_APP, filenames[0]))
    fig = get_graph_figure(img)
    return [dcc.Graph(
        id='graph-picture',
        figure=fig,
        config={
            # 'displayModeBar': True,
            'editable': False,
            'scrollZoom': False,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                'autoScale2d', 'resetScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian',
                'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
                'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews',
                'toggleSpikelines', 'resetViewMapbox'
            ]
        }
    )]


@du.callback(
    Output("div-graph-normal", "children"),
    id='uploader-normal',
)
def update_graph_normal(filenames):
    path_img = os.path.join(PATH_APP, filenames[0])
    fig = base64.b64encode(open(path_img, 'rb').read())

    return [html.Img(
        id='graph-normal',
        src='data:image/png;base64,{}'.format(fig.decode()),
        height="100%",
        width="100%",
        style={'margin': 'auto'}
    )]


@du.callback(
    Output("div-graph-envmap", "children"),
    id='uploader-envmap',
)
def update_graph_envmap(filenames):
    path_img = os.path.join(PATH_APP, filenames[0])
    if filenames[0].endswith('exr'):
        img = utils.exr2ldr(path_img)
    elif filenames[0].endswith('hdr'):
        img = utils.hdr2ldr(path_img)
    else:
        img = cv2.imread(path_img)
    # fig = base64.b64encode(open(path_img, 'rb').read())
    _, img = cv2.imencode(".png", img)
    fig = base64.b64encode(img.tobytes())

    return [html.Img(
        id='graph-envmap',
        src='data:image/png;base64,{}'.format(fig.decode()),
        height="100%",
        width="100%",
        style={'margin': 'auto'}
    )]


@app.callback(
    [Output("input-obj-pos-x", "value"), Output("input-obj-pos-y", "value")],
    [Input("graph-picture", "clickData")],
    prevent_initial_call=True,
)
def update_axios_output(click_data):
    if click_data:
        return click_data['points'][0]['x'], click_data['points'][0]['y']
    else:
        return '0000', '0000'


@app.callback(
    [
        Output("loading-render", "children"),
        Output("debug-output", "children"),
        Output("div-graph-output", "children"),
    ],
    [
        Input("bt-start-render", "n_clicks")
    ],
    [
        State("uploader", "fileNames"),
        State("uploader-normal", "fileNames"),
        State("uploader-envmap", "fileNames"),
        State("input-envmap-scale", "value"),
        State("dropdown-obj-shape", "value"),
        State("input-obj-translate", "value"),
        State("input-obj-color", "value"),
        State("input-obj-size", "value"),
        State("input-obj-pos-x", "value"),
        State("input-obj-pos-y", "value")
    ]
)
def on_bt_start_render_click(n, path_img, path_normal, path_envmap, envmap_scale, obj_shape, obj_trans_scale, obj_color, obj_size, obj_x, obj_y):
    if n is None:
        return ["Start Render"], [], []

    path_img = "" if path_img is None else osp.join(PATH_APP, "data", "inputs", path_img[0])
    path_normal = "" if path_normal is None else osp.join(PATH_APP, "data", "inputs", path_normal[0])
    path_envmap = "" if path_envmap is None else osp.join(PATH_APP, "data", "envmap", path_envmap[0])
    string_debug = (
        "n: {}; path_img: {}; path_normal: {}; path_envmap: {}; envmap_scale: {}, obj_shape: {}; obj_trans_scale: {}; obj_color: {}; obj_size: {}; obj_x: {}; obj_y: {};".format(
            n, path_img, path_normal, path_envmap, envmap_scale, obj_shape, obj_trans_scale, obj_color, obj_size, obj_x, obj_y))

    if not (path_img != "" and path_normal != "" and path_envmap != ""):
        return ["Start Render"], [string_debug], []

    image = Image.open(path_img)
    height, width = image.size[1], image.size[0]

    obj2D = np.asarray([obj_x, obj_y])
    quad2D = utils.point2quadrangle(height, width, point=obj2D, offset=10)
    mask2D = render.generatePlaneMask(height, width, quad2D)
    vn3D, v3D, _ = render.generatePlane3D(height, width, quad2D, mask2D, path_normal=path_normal, path_plane="")
    obj_v3D, _ = render.generateObject3D(height, width, point=obj2D, quad=quad2D, quad_vn=vn3D, quad_v=v3D)
    plane2D = utils.point2quadrangle(height, width, point=obj2D, offset=50)
    plane_mask2D = render.generatePlaneMask(height, width, plane2D)
    _, plane3D, _ = render.generatePlane3D(height, width, plane2D, plane_mask2D, plane_vn=vn3D, path_normal=path_normal, path_plane=config.PATH_OUT_PLANE)

    render.generateSceneXML(
        height, width, vn3D,
        path_env=path_envmap,
        path_albedo=config.PATH_DATA_ALBEDO,
        path_rough=config.PATH_DATA_ROUGH,
        path_plane=config.PATH_OUT_PLANE,
        path_obj=config.PATH_DATA_OBJECT_BUNNY if obj_shape == 1 else config.PATH_DATA_OBJECT,
        path_out=config.PATH_OUT_XML,
        envmap_scale=envmap_scale,
        obj_diffuse="#ffffff",
        obj_scale=obj_size,
        obj_translate=obj_trans_scale * obj_v3D,
        sample=1024,
        is_hdr=False,
    )
    render.differentialRender(path_mitsuba=config.PATH_MITSUBA, path_img_in=path_img, path_img_out=config.PATH_IMAGE_OUT, path_out=config.PATH_OUT)

    output = Image.open(config.PATH_IMAGE_OUT)
    fig_out = get_graph_figure(output)

    return ["Render Done"], [string_debug], [dcc.Graph(id='graph-output', figure=fig_out,
                                                       config={
                                                           # 'displayModeBar': True,
                                                           'editable': False,
                                                           'scrollZoom': False,
                                                           'displaylogo': False,
                                                           'modeBarButtonsToRemove': [
                                                               'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                                                               'autoScale2d', 'resetScale2d',
                                                               'hoverClosestCartesian', 'hoverCompareCartesian',
                                                               'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
                                                               'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews',
                                                               'toggleSpikelines', 'resetViewMapbox'
                                                           ]
                                                       })]
