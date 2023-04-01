from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from views.index import index_page
from server import app

app.layout = html.Div([
    dcc.Location(id='url'),
    dbc.Navbar(
        # dark=True,
        color="#EDEFEB",
        style={'margin-top': '10px'},
        children=[
            html.A(
                href="https://vrlab.buaa.edu.cn/",
                style={'text-decoration': 'none'},
                children=dbc.Row(
                    align="center",
                    # no_gutters=True,
                    children=[
                        dbc.Col(
                            width={'size': 2, 'offset': 3},
                            children=html.Img(
                                src=app.get_asset_url("home.png"),
                                height="60px"
                            ),
                        ),
                        dbc.Col(
                            width={'size': 2, 'offset': 2},
                            children=dbc.NavbarBrand(
                                style={'color': 'black', 'font-size': '1.6em', 'font-weight': 'bold', 'font-family': 'ui-monospace'},
                                className="ml-1 mt-1",
                                children="Wangxc",
                            ),
                        )
                    ],
                ),

            ),
            dbc.Row(
                align="center",
                # no_gutters=True,
                className="ml-auto mr-4 flex-nowrap mt-6 md-0",
                children=[
                    dbc.Col(
                        children=dbc.NavLink(
                            style={'color': 'black', 'font-size': '1.2em', 'font-weight': 'bold', 'font-family': 'ui-monospace'},
                            active="exact",
                            className="mt-1",
                            children='Home',
                            href='/',
                        )
                    ),
                ],
            )
        ],
    ),
    html.Div(
        id='page-content',
        style={'flex': 'auto'}
    ),
])


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/':
        return index_page

    return html.H1('您访问的页面不存在！')


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
