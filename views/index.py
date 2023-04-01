import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash_daq as daq

from views.index_callbks import *

index_page = html.Div([
    dbc.Container(
        fluid=True,
        style={'width': '95%', 'margin-top': '5px'},
        children=[
            dbc.Row([
                dbc.Col(
                    width=3,
                    children=[
                        html.Div(
                            className='box',
                            style={
                                'width': '100%',
                                'height': '280px',
                                'margin-bottom': '20px',
                                'padding-top': '15px',
                                'padding-bottom': '15px'
                            },
                            children=[
                                html.Div(
                                    style={'height': '50px'},
                                    children=dbc.Row(
                                        justify='center',
                                        align='center',
                                        style={'height': '100%'},
                                        children=[dbc.Col(
                                            id='div-graph-normal',
                                            style={'height': '100%'},
                                            children=du.Upload(
                                                id='uploader-normal',
                                                text='Drag/Drop here to upload normal!',
                                                text_completed='',
                                                filetypes=['png'],
                                                upload_id='inputs',
                                                default_style={
                                                    'height': '100%',
                                                    'minHeight': 1, 'lineHeight': 1,
                                                    'textAlign': 'center',
                                                    'outlineColor': '#ea8f32',
                                                    'font-family': 'Open Sans',
                                                    'font-size': '15px',
                                                    'font-weight': '500',
                                                }
                                            ),
                                        )]
                                    )
                                ),
                                html.Div(style={'height': '13px'}),
                                html.Div(
                                    style={'height': '180px'},
                                    children=dbc.Row(
                                        justify='center',
                                        align='center',
                                        style={'height': '100%'},
                                        children=[dbc.Col(
                                            id='div-graph-normal',
                                            style={'height': '100%'},
                                            children=html.Img(id="graph-normal"),
                                        )]
                                    )
                                ),
                            ]
                        ),
                        html.Div(
                            className='box',
                            style={
                                'width': '100%',
                                'margin-bottom': '20px',
                                'padding-top': '10px',
                                'padding-bottom': '15px'
                            },
                            children=[
                                dbc.Row(
                                    align="center",
                                    children=[
                                        dbc.Col(
                                            width=6,
                                            children=[dbc.Row(
                                                align="center",
                                                justify="center",
                                                children=[
                                                    dbc.Col([
                                                        dbc.FormText("shape of object"),
                                                        dcc.Dropdown(
                                                            id="dropdown-obj-shape",
                                                            options=[
                                                                {"label": "rabbit", "value": 1},
                                                            ],
                                                            value=1
                                                        )
                                                    ], width=12),
                                                ]
                                            )]
                                        ),
                                        dbc.Col(
                                            width=6,
                                            children=[dbc.Row(
                                                align="center",
                                                justify="center",
                                                children=[
                                                    dbc.Col([
                                                        dbc.FormText("translate scale"),
                                                        dbc.Input(
                                                            id='input-obj-translate',
                                                            type='number',
                                                            value=0.97,
                                                            min=0.0, max=10.0, step=0.01
                                                        ),
                                                    ], width=12),
                                                ]
                                            )]
                                        ),
                                    ],
                                ),
                                html.Div(style={'height': '2px'}),
                                dbc.Row(
                                    align="center",
                                    children=[
                                        dbc.Col(
                                            width=6,
                                            children=[dbc.Row(
                                                align="center",
                                                justify="center",
                                                children=[
                                                    dbc.Col([
                                                        dbc.FormText("size of object"),
                                                        dbc.Input(
                                                            id='input-obj-size',
                                                            type='number',
                                                            value=0.6,
                                                            min=0, max=10, step=0.1
                                                        ),
                                                    ], width=12),
                                                ]
                                            )]
                                        ),
                                        dbc.Col(
                                            width=6,
                                            children=[dbc.Row(
                                                align="center",
                                                justify="center",
                                                children=[
                                                    dbc.Col([
                                                        dbc.FormText("color of object"),
                                                        dbc.Input(
                                                            id='input-obj-color',
                                                            value="#ffffff",
                                                        ),
                                                    ], width=12),
                                                ]
                                            )]
                                        )
                                    ],
                                ),
                                html.Div(style={'height': '2px'}),
                                dbc.Row(
                                    align="center",
                                    children=[
                                        dbc.Col(
                                            width=6,
                                            children=[dbc.Row(
                                                align="center",
                                                justify="center",
                                                children=[
                                                    dbc.Col([
                                                        dbc.FormText("x of position"),
                                                        dbc.Input(
                                                            id='input-obj-pos-x',
                                                            type='number',
                                                            value=0,
                                                            min=0, max=8048, step=1
                                                        ),
                                                    ], width=12),
                                                ]
                                            )]
                                        ),
                                        dbc.Col(
                                            width=6,
                                            children=[dbc.Row(
                                                align="center",
                                                justify="center",
                                                children=[
                                                    dbc.Col([
                                                        dbc.FormText("y of position"),
                                                        dbc.Input(
                                                            id='input-obj-pos-y',
                                                            type='number',
                                                            value=0,
                                                            min=0, max=8048, step=1
                                                        ),
                                                    ], width=12),
                                                ]
                                            )]
                                        )
                                    ],
                                )
                            ]
                        ),

                    ]
                ),
                dbc.Col(
                    width=6,
                    children=[
                        html.Div(
                            className='box',
                            style={
                                'width': '100%',
                                # 'height': '80%',
                                'padding-top': '10px',
                                # 'padding-bottom': '15px',
                                'display': 'flex',
                                'align-items': 'center'
                            },
                            children=[
                                html.Div(
                                    style={'width': '100%', 'height': '100%'},
                                    children=dcc.Tabs(
                                        id='tabs',
                                        value='1',
                                        style={
                                            'height': '5px',
                                            # 'width': '80%',
                                            'border': 0,
                                            'margin': '15px',
                                            'margin-top': '15px',
                                            'margin-bottom': '5px'
                                        },
                                        children=[
                                            dcc.Tab(
                                                id="tab-image",
                                                value="1",
                                                style={
                                                    'padding': '0px',
                                                    'border': 0,
                                                    'border-radius': '5px',
                                                    'border-top-right-radius': '0px',
                                                    'border-bottom-right-radius': '0px',
                                                    'backgroundColor': '#edefeb'
                                                },
                                                selected_style={
                                                    'padding': '0px',
                                                    'border': 0,
                                                    'border-radius': '5px',
                                                    'border-top-right-radius': '0px',
                                                    'border-bottom-right-radius': '0px',
                                                    'backgroundColor': '#4b9072'
                                                },
                                                children=[
                                                    dbc.Container(dcc.Graph(id='graph-picture'), id='div-graph-picture')
                                                ],
                                            ),
                                            dcc.Tab(
                                                id="tab-output",
                                                value="2",
                                                style={
                                                    'padding': '0px',
                                                    'border': 0,
                                                    'border-radius': '5px',
                                                    'border-top-right-radius': '0px',
                                                    'border-bottom-right-radius': '0px',
                                                    'backgroundColor': '#edefeb'
                                                },
                                                selected_style={
                                                    'padding': '0px',
                                                    'border': 0,
                                                    'border-radius': '5px',
                                                    'border-top-right-radius': '0px',
                                                    'border-bottom-right-radius': '0px',
                                                    'backgroundColor': '#4b9072'
                                                },
                                                children=[
                                                    dbc.Container(dcc.Graph(id='graph-output'), id='div-graph-output')
                                                ],
                                            ),
                                        ])
                                ),
                            ],
                        )
                    ]
                ),
                dbc.Col(
                    width=3,
                    children=[
                        html.Div(
                            className='box',
                            style={
                                'width': '100%',
                                # 'height': '207px',
                                'margin-bottom': '20px',
                                'padding-top': '15px',
                                'padding-bottom': '15px'
                            },
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row([
                                            du.Upload(
                                                id='uploader',
                                                text='Drag/Drop here to upload image!',
                                                text_completed='',
                                                filetypes=['png'],
                                                upload_id='inputs',
                                                default_style={
                                                    'height': '100%',
                                                    'minHeight': 1, 'lineHeight': 1,
                                                    'textAlign': 'center',
                                                    'outlineColor': '#ea8f32',
                                                    'font-family': 'Open Sans',
                                                    'font-size': '15px',
                                                    'font-weight': '500',
                                                }
                                            ),
                                        ]),
                                        # dbc.Row(
                                        #     html.Div(style={"height": "13px"})
                                        # ),
                                    ], width=11),
                                ], justify="center")
                            ]
                        ),
                        html.Div(
                            className='box',
                            style={
                                'width': '100%',
                                'height': '320px',
                                'margin-bottom': '20px',
                                'padding-top': '15px',
                                'padding-bottom': '15px'
                            },
                            children=[
                                html.Div(
                                    style={'height': '50px'},
                                    children=dbc.Row(
                                        justify='center',
                                        align='center',
                                        style={'height': '100%'},
                                        children=[dbc.Col(
                                            id='div-graph-envmap',
                                            style={'height': '100%'},
                                            children=du.Upload(
                                                id='uploader-envmap',
                                                text='Drag/Drop here to upload envmap!',
                                                text_completed='',
                                                filetypes=['png', 'exr', 'hdr'],
                                                upload_id='envmap',
                                                default_style={
                                                    'height': '100%',
                                                    'minHeight': 1, 'lineHeight': 1,
                                                    'textAlign': 'center',
                                                    'outlineColor': '#ea8f32',
                                                    'font-family': 'Open Sans',
                                                    'font-size': '15px',
                                                    'font-weight': '500',
                                                }
                                            ),
                                        )]
                                    )
                                ),
                                html.Div(style={'height': '13px'}),
                                html.Div(
                                    style={'height': '150px'},
                                    children=dbc.Row(
                                        justify='center',
                                        align='center',
                                        style={'height': '100%'},
                                        children=[dbc.Col(
                                            id='div-graph-envmap',
                                            style={'height': '100%'},
                                            children=html.Img(id="graph-envmap"),
                                        )]
                                    )
                                ),
                                html.Div(style={'height': '13px'}),
                                html.Div(
                                    style={'height': '50px'},
                                    children=dbc.Row(
                                        align="center",
                                        children=[
                                            dbc.Col(
                                                width=6,
                                                children=[dbc.Row(
                                                    align="center",
                                                    justify="center",
                                                    children=[
                                                        dbc.Col([
                                                            dbc.FormText("scale of envmap"),
                                                            dbc.Input(
                                                                id='input-envmap-scale',
                                                                type='number',
                                                                value=1,
                                                                min=0
                                                            ),
                                                        ], width=12),
                                                    ]
                                                )]
                                            ),
                                        ],
                                    ),
                                )
                            ]
                        ),
                        html.Div(
                            className='box',
                            style={
                                'width': '100%',
                                'margin-bottom': '20px',
                                'padding-top': '16px',
                                'padding-bottom': '16px'
                            },
                            children=
                            html.Div([
                                # dbc.Container([
                                dbc.ButtonGroup([
                                    dbc.Button([dbc.Spinner(size="sm", children=[html.Div(id="loading-render")])],
                                               color="success", outline=False, id='bt-start-render',
                                               style={'background-color': '#4b9072', 'color': 'white'})
                                ], style={'width': '100%'}),
                                # ], fluid=True, style={'margin-top': '5px'}),
                            ]),
                        ),
                    ]
                )
            ]),
            dbc.Row([
                dbc.Col(width=12, children=[
                    html.Div(
                        className='box',
                        style={
                            'width': '100%',
                            'margin-bottom': '20px',
                            'padding-top': '15px',
                            'padding-bottom': '15px'
                        },
                        children=html.Div(id="debug-output"),
                    ),
                ])

            ])
        ]
    )
])
