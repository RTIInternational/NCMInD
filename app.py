import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json
import pandas as pd
import plotly.graph_objs as go

from src.analyze import Analyze
from src.state import *

# ----- Standard App Creation
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# ----- Setup the App Layout
app.layout = html.Div(children=[

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='scenario_dir', style={'display': 'none'}),

    html.H1("Orange County: NCMiND Demo."),

    html.Div(
        [
            html.H6("Update Parameters:"),
            html.Div(
                [
                    html.P(id="colonization_text", style={'textAlign': 'left'}),
                    dcc.Slider(
                        id='colonization_probability',
                        min=.01, max=.75, value=.02, step=.0001,
                        marks={.1: "10%",
                               .25: "25%",
                               .50: "50%",
                               .75: "75%"}
                    )
                ],
                className="three columns"
            ),

            html.Div(
                [
                    html.P(id="hospital_probability_text", style={'textAlign': 'left'}),
                    dcc.Slider(
                        id='hospital_probability',
                        min=.50, max=4.00, step=.50, value=1,
                        marks={.50: "-50%",
                               1.0: "+0%", 1.50: "+50%",
                               2.0: "+100%", 2.5: "+150%",
                               3.0: "+200%", 3.5: "+250%",
                               4.0: "+300%"}
                    )
                ],
                className="three columns"
            ),

            dcc.Interval(
                        id='interval-component',
                        interval=1*1500,  # in milliseconds
                        n_intervals=0
                    )
        ],
        className="row",
        style={"marginBottom": "50"}
    ),

    html.Div(
        [
            dcc.Tabs(id="tabs",
                     children=[
                         dcc.Tab(label='Explore Agent Movement', children=[
                             html.Div([
                                 dcc.Graph(id='location_counts', style={"height": "600px"})
                             ], className="four columns"),
                             html.Div([
                                 dcc.Graph(id='location_graph', style={'height': "600px"})
                             ], className="seven columns")
                         ]),
                         dcc.Tab(label='Explore CDI Risk', children=[
                             html.Div([
                                 dcc.Graph(id='risk_counts', style={"height": "350px"}),
                                 dcc.Graph(id='risk_metrics', style={'height': "350px"})
                             ], className="four columns"),
                             html.Div([
                                 dcc.Graph(id='risk_graph', style={'height': "600px"})
                             ], className="seven columns")
                         ])
                     ])
        ],
        className="row"
    )

], style={'vertical-align': 'middle'})


# ----- Likelihood of going to a hospital:
@app.callback(
    Output('colonization_text', 'children'),
    [Input('colonization_probability', 'value')]
)
def update_output_div(input_value):
    # Save the user input
    with open("NCMIND/demo/default/parameters2.json") as f:
        params = json.load(f)
    params['risk']['colonization_recovery'] = input_value

    with open('NCMIND/demo/default/parameters2.json', 'w') as outfile:
        json.dump(params, outfile)

    return 'Chance to recover from colonization: {:.0%}'.format(input_value)


# ----- Likelihood of going to a hospital:
@app.callback(
    Output('hospital_probability_text', 'children'),
    [Input('hospital_probability', 'value')]
)
def update_output_div(input_value):
    # Save the user input
    with open("NCMIND/demo/default/parameters2.json") as f:
        params = json.load(f)
    params['demo']['community_adjustment'] = input_value

    with open('NCMIND/demo/default/parameters2.json', 'w') as outfile:
        json.dump(params, outfile)

    if input_value >= 1:
        input_value = input_value - 1
        return 'Community to Hospital Probability: +{:.0%}'.format(input_value)
    return 'Community to Hospital Probability: {:.0%}'.format(input_value)


# ---- Location Graphics
@app.callback(
    Output('location_graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def location_graph(n_intervals):
    if n_intervals or True:
        a = Analyze(exp_dir="NCMIND/demo", scenario_dir='default', run_dir="")
        return a.make_location_graph_combined(filename='')


@app.callback(
    Output('location_counts', 'figure'),
    [Input('interval-component', 'n_intervals')])
def location_counts(n_intervals):
    if n_intervals or True:
        a = Analyze(exp_dir="NCMIND/demo", scenario_dir='default', run_dir="")

        aa = a.count_by_x('Location')
        df = pd.DataFrame(aa.iloc[:, -1])
        df.index = [a.map[a.locations(i).name] for i in df.index]
        df = df.reset_index()
        df.columns = ['Location', 'Count of People']
        df['Count of People'] = df.apply(lambda x: "{:0,}".format(x['Count of People']), axis=1)

        trace = go.Table(
            header=dict(values=['Location', 'Count of People: Day ' + str(aa.shape[1])],
                        line=dict(color='#7D7F80'),
                        fill=dict(color='#a1c3d1'),
                        align=['left'] * 5),
            cells=dict(values=[df['Location'], df['Count of People']],
                       line=dict(color='#7D7F80'),
                       fill=dict(color='#EDFAFF'),
                       align=['left'] * 5))
        layout = dict()
        data = [trace]

        return dict(data=data, layout=layout)


# ----- CDI Graphics
@app.callback(
    Output('risk_graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def risk_graph(n_intervals):
    if n_intervals or True:
        a = Analyze(exp_dir="NCMIND/demo", scenario_dir='default', run_dir="")
        return a.make_cdiff_graph(filename='')


@app.callback(
    Output('risk_counts', 'figure'),
    [Input('interval-component', 'n_intervals')])
def risk_counts(n_intervals):
    if n_intervals or True:
        a = Analyze(exp_dir="NCMIND/demo", scenario_dir='default', run_dir="")

        aa = a.count_by_x('Risk')
        df = pd.DataFrame(aa.iloc[:, -1])
        df.index = [a.cdiff_map[CDIFFState(i).name] for i in df.index]
        df = df.reset_index()
        df.columns = ['CDI Risk', 'Count of People']
        df['Count of People'] = df.apply(lambda x: "{:0,}".format(x['Count of People']), axis=1)
        df = df.drop(4)

        trace = go.Table(
            header=dict(values=['CDI Risk', 'Count of People: Day ' + str(aa.shape[1])],
                        line=dict(color='#7D7F80'),
                        fill=dict(color='#a1c3d1'),
                        align=['left'] * 5),
            cells=dict(values=[df['CDI Risk'], df['Count of People']],
                       line=dict(color='#7D7F80'),
                       fill=dict(color='#EDFAFF'),
                       align=['left'] * 5))
        layout = dict()
        data = [trace]

        return dict(data=data, layout=layout)


@app.callback(
    Output('risk_metrics', 'figure'),
    [Input('interval-component', 'n_intervals')])
def risk_metrics(n_intervals):
    if n_intervals or True:
        a = Analyze(exp_dir="NCMIND/demo", scenario_dir='default', run_dir="")
        aa = a.count_by_x('Risk')

        v = []
        associated = a.associated_cdi(seventy_plus=False)
        # onset = a.onset_cdi(seventy_plus=False)
        v.append(associated[1][0])  #
        v.append(associated[1][1])  #
        v.append(a.cdi_deaths())
        v.append(a.prevalence(risks=[CDIFFState.HR], last_9=False))
        v = [round(item, 2) for item in v]
        names = ['Community Associated - CDI', 'Hospital Associated - CDI', 'Total CDI Deaths', 'Antibiotic Prevalence']

        trace = go.Table(
            header=dict(values=['Metric', 'Metric Value: Day ' + str(aa.shape[1])],
                        line=dict(color='#7D7F80'),
                        fill=dict(color='#a1c3d1'),
                        align=['left'] * 5),
            cells=dict(values=[names, v],
                       line=dict(color='#7D7F80'),
                       fill=dict(color='#EDFAFF'),
                       align=['left'] * 5))
        layout = dict()
        data = [trace]

        return dict(data=data, layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
