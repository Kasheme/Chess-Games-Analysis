import dash 
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import no_update
import pandas as pd
import numpy as np
import pathlib

# Connect to main app.py file
from chess_games_app import app
from chess_games_app import server

# Connect to app pages
from apps import chess_games_about

# Read the top games data into a pandas dataframe
top_games = pd.read_csv(r"https://raw.githubusercontent.com/Kasheme/Chess-Games-Analysis/master/datasets/top%20games.csv", encoding='iso-8859-1')

# Read the games data into a pandas dataframe and perform necessary cleansing
games = pd.read_csv("https://raw.githubusercontent.com/Kasheme/Chess-Games-Analysis/master/datasets/games.csv", encoding='iso-8859-1')
games = games.drop(columns={'ï»¿'})
games = games.rename(columns={'opening_classification':'opening','opening_name':'variation'})

"""
==============
Markdown text
"""

footer = html.Div(
                dcc.Markdown(
                    """
                    This information is intended solely as general information for educational purposes only.
                    Any questions, feedback or suggestions please don't hesitate to get in touch:
                    [Email](mailto:kasheme.walton@outlook.com?)
                    You can view all project content here: [Github](https://github.com/Kasheme/Chess-Games-Analysis)
                    
                    """,
                highlight_config={'theme':'dark'}),
                className="p-2 mt-5 bg-primary text-white small",
            )
"""
=====================================================
Create helper functions for computing layouts and data

"""
# Function to create a card template
def create_card(df, title, move, color, inverse=True, fontcolor='white'):
    return dbc.Card(
                dbc.CardBody(
                    [
                            html.H6(title, className="card-title",
                                     style={'font-size': 14, 'textAlign': 'center',
                                            'color': fontcolor}),
                            html.H6(df[move][0], className="card-text",
                                    style={'font-size': 24, 'textAlign': 'center',
                                           'color': fontcolor})
                    ]
               ),
               color=color, inverse=inverse, className="w-20")

# Compute graphs data for openings report: df = top_games, df2 = games, value = opening
def compute_data_choice1(df, df2, value):
    # Select data
    results_df = df[df['opening'] == value].groupby(by='opening', axis=0).mean().round(0).astype(np.int64).transpose().loc[['%draw', '%mate','%resign', '%outoftime'],:]
    winner_df = df[df['opening'] == value].groupby(by='opening', axis=0).mean().round(0).astype(np.int64).transpose().loc[['%draw', '%white_wins', '%black_wins'],:]
    filtered_df = df[df['opening'] == value].iloc[:,[0,1,2,3,4,5]]
    
    scatter_df = df2[df2['opening'] == value].iloc[:,[0,2,8,9]]
    scatter_df['frequency'] = 1
    scatter_df = scatter_df.pivot_table(index='variation', aggfunc={'turns': np.mean, 'opening_ply': np.mean,'frequency': np.sum}).astype(int)
    scatter_df = scatter_df.rename(columns={'turns': 'avg_turns', 'opening_ply': 'avg_opening_ply'})
    return results_df, winner_df, filtered_df, scatter_df

# Compute graphs data for variations report: df = top_games, df2 = games, value = variation
def compute_data_choice2(df, df2, value):
    # Select data
    results_df = df[df['variation'] == value].groupby(by='variation', axis=0).mean().round(0).astype(np.int64).loc[:,['%draw', '%mate','%resign', '%outoftime']].transpose()
    winner_df = df[df['variation'] == value].groupby(by='variation', axis=0).mean().round(0).astype(np.int64).loc[:,['%draw', '%white_wins', '%black_wins']].transpose()
    hist_df = df2[df2['variation'] == value].iloc[:,[0,1,2,3,9]]
    
    rated_df = df2[df2['variation'] == value].loc[:,['rated', 'variation']]
    rated_df['count'] = 1
    rated_df = rated_df.groupby(by='rated', axis=0).sum()
    
    return results_df, winner_df, hist_df, rated_df

def compute_grouped_boxplot(df, df2, value):
    
    # Compute grouped box plot: df = top_games, df2 = games, value = opening
    groupedbox_df = df[df['opening'] == value]
    x_data = list(groupedbox_df['variation'].unique())
    
    y0 = np.array(df2[df2['variation'] == x_data[0]].iloc[:,2])
    y1 = np.array(df2[df2['variation'] == x_data[1]].iloc[:,2])
    y2 = np.array(df2[df2['variation'] == x_data[2]].iloc[:,2])
    y3 = np.array(df2[df2['variation'] == x_data[3]].iloc[:,2])
    y4 = np.array(df2[df2['variation'] == x_data[4]].iloc[:,2])
    y5 = np.array(df2[df2['variation'] == x_data[5]].iloc[:,2])
    y6 = np.array(df2[df2['variation'] == x_data[6]].iloc[:,2])
    y7 = np.array(df2[df2['variation'] == x_data[7]].iloc[:,2])
    y8 = np.array(df2[df2['variation'] == x_data[8]].iloc[:,2])
    y9 = np.array(df2[df2['variation'] == x_data[9]].iloc[:,2])
    
    y_data = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]
    
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
             'rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
             'rgba(255, 65, 54, 0.5)']

    fig = go.Figure()
    
    for xd, yd, cls in zip(x_data, y_data, colors):
        fig.add_trace(go.Box(
                y=yd,
                name=xd,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                fillcolor=cls,
                marker_size=2,
                line_width=1)
            )
        
    fig.update_layout(
        title='Comparitive statistical distribution of number of turns for top10 variations',
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=5,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
        ),
        margin=dict(
            l=40,
            r=30,
            b=80,
            t=100,
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        showlegend=False,
        font=dict(color='white'),
        width=1235,
        height=500
    )
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig


"""
===========================
Layout section of dashboard
"""

app.layout =  html.Div(children=[html.Div([
                                # page Heading
                                html.H1('Chess Games Analysis Interactive Dashboard',
                                        style={'textAlign': 'center', 'color': 'black', 'font-size': 36,
                                               'margin-bottom': '0.25em', 'width':'100%'},
                                        className="text-center bg-primary text-white p-2"),
                                html.Div([
                                    # link to about page
                                    dcc.Link('About', href='/apps/chess_games_about'),
                                    dcc.Link('Dashboard', href='/chess_games_dashboard')
                                    ], className="row", style={'marign-left':'10em'}
                                    ),
                                    dcc.Location(id='url', refresh='False'),
                                    html.Div(id='page-content', children=[ ]),
                                ]),
                                # "title" for First 10 moves display
                                dbc.Row([
                                    dbc.Col(
                                        html.Div("Most common opening moves",
                                                 style={'font': "Open Sans", 'color': 'white', 'font-size': 18}),
                                                 style={'textAlign': 'right', 'margin-bottom': '0.5em'},    
                                                 width={'size': 6, 'offset': 3}
                                            ),
                                    # Button and tooltip for first 10 moves display
                                    dbc.Col(
                                        html.Div([
                                             dbc.Button(
                                                    "i",
                                                    id="tooltip_button",
                                                    className="me-1",
                                                    color="dark", 
                                                    size='sm',
                                                    n_clicks=0
                                                        ),
                                             dbc.Tooltip(
                                                    "For the variations report, this will display the first 10 most occuring moves in the dataset. In chess, only 1-2 moves determine which opening has been used, so only the first 2 most used moves will be displayed for the openings report.",
                                                    target="tooltip_button",
                                                    placement='right'
                                                    )
                                             ])
                                        )
                                    ]),
                        # outer division starts                                
                        html.Div([
                            # Add first division
                            html.Div([
                                    # Create a division for adding dropdown helper text for selected report type
                                    html.Div([    
                                        html.H2('Report Type:  ', style={'margin-right': '6.4em', 'margin-left': '2em',
                                                                         'font-size': 28}),
                                        ]
                                    ),
                                        # First dropdown for selecting a report
                                        dcc.Dropdown(id='report_type',
                                                     options=[
                                                             {'label': 'Openings Report', 'value': 'OPT1'},
                                                             {'label': 'Variations Report', 'value': 'OPT2'}
                                                             ],
                                                     placeholder='Select a report type',
                                                     style={'width':'50%', 'padding': '3px', 
                                                            'font-size': '20px', 'text-align-last': 'left'}),
                                   # First 5 moves card sequence Layout
                                   html.Div([
                                       dbc.Row(
                                           [
                                            dbc.Col(html.Div([
                                                 html.Div(id='move1_id'),
                                                 ], className = "create_move1_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move2_id'),
                                                 ], className = "create_move2_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move3_id'),
                                                 ], className = "create_move3_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move4_id'),
                                                 ], className = "create_move4_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move5_id'),
                                                 ], className = "create_move5_card", style={'text-align': 'center'})),
                                            
                                           ], style={'margin-left': '-17em'}, className="g-0", id='cards1'
                                         )
                                       
                                         ], style={'display': 'flex', 'flex-direction': 'row'}
                                       ), 
                                   
                                   ], style={'display': 'flex', 'flex-direction': 'row'}),
                            
                            
                                    # Next division for opening classification, dropdown and next 5 moves
                                    html.Div([
                                        # Division for helper text
                                        html.Div([
                                            html.H2('Opening Classification: ', style={'margin-right': '1.75em', 'margin-left': '2em', 'font-size': 28}),
                                            ]
                                    ),
                                        # Openings dropdown
                                        dcc.Dropdown(
                                            ['English opening', 'French defence', "King's pawn game", "Philidor's defence", 'Polish (Sokolsky)  opening',
                                             "Queen's pawn", "Queen's pawn game", 'Ruy Lopez (Spanish opening)', 
                                             'Scandinavian (centre counter) defence', 'Sicilian defence'],
                                            id='class_dropdown',
                                            placeholder='Select an opening classification',
                                            style={'width':'50%', 'padding': '3px', 'font-size': '20px'}),
                                    
                                    # Card sequence Layout
                                    html.Div([
                                       dbc.Row(
                                           [
                                            dbc.Col(html.Div([
                                                 html.Div(id='move6_id'),
                                                 ], className = "create_move6_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move7_id'),
                                                 ], className = "create_move7_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move8_id'),
                                                 ], className = "create_move8_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move9_id'),
                                                 ], className = "create_move9_card", style={'text-align': 'center'})),
                                            dbc.Col(html.Div([
                                                 html.Div(id='move10_id'),
                                                 ], className = "create_move10_card", style={'text-align': 'center'})),
                                            
                                           ], style={'margin-left': '-17em'}, className="g-0", id='cards2'
                                         )
                                       
                                         ], style={'display': 'flex', 'flex-direction': 'row'}
                                       ), 

                                    ], style={'display': 'flex', 'flex-direction': 'row'}),

                                html.Div(children=[
                                    html.Div([
                                        # helper text for selecting variation
                                        html.H2('Opening Variation: ', style={'margin-right': '2.5em', 'margin-left': '2em',
                                                                                'font-size': 22},
                                                id='variation_text'),
                                        ]
                                    ),
                                        # Dropdown for selecting a variation
                                        dcc.Dropdown(id='variation_dropdown',
                                                     multi=False,
                                                     style={'width':'50%', 'padding': '3px', 'font-size': '20px'},
                                                     options = [],
                                                     placeholder="Select an opening variation"
                                                     )
                                        ], style={'display': 'flex', 'flex-direction': 'row'}),
                                    
                                   
                                        # 1st segment
                                        html.Div([
                                            html.Div([ ], id='plot1'),
                                            html.Div([ ], id='plot2'),
                                            html.Div([ ], id='plot3')
                                            ], style={'display': 'flex'}),
                                        
                                        # 2nd segment
                                        html.Div([
                                             html.Div([ ], id='plot4'),
                                             html.Div([ ], id='plot5')
                                            ], style={'display': 'flex'}),
                                        
                                        # add footer
                                        dbc.Row(dbc.Col(footer))
                                            
                                    ])
                                    # outer division ends
                                ])
                                # layout ends
                                
"""
==================
Callback functions
"""

""" Callback function to go to about page if link selected"""
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/apps/chess_games_about':
        return chess_games_about.layout
    else:
        pass
                                
""" Callback function that disables the variations dropdown if the Openings Report is selected """
@app.callback(
    Output('variation_dropdown', 'style'),
    Output('variation_text', 'style'),
    Input('report_type', 'value'))                                
def disable_dropdown(report_type):
    if report_type == 'OPT1':
        return {'display': 'none'}, {'display': 'none'}
    else:
        return [
                {'width':'50%', 'padding': '3px', 'font-size': '20px'},
                {'margin-right': '3.8em', 'margin-left': '2em', 'font-size': 28}    
                ]

""" Callback function that returns options for the variations dropdown """
@app.callback(
    Output('variation_dropdown', 'options'),
    Input('class_dropdown', 'value'))
def get_options(class_dropdown):
    var_df = top_games[top_games['opening'] == class_dropdown]
    variation_options = [{'label': i, 'value': i} for i in var_df['variation'].unique()]
    variation_options.append({'label' :'All', 'value' : 'All'})
    return variation_options
                            
""" Callback function that returns values for the variations dropdown """
@app.callback(
    Output('variation_dropdown', 'value'),
    Input('variation_dropdown', 'options'))
def get_value(variation_dropdown):
    return [v['value'] for v in variation_dropdown[0:1]]


""" Callback function that returns top10 moves if a variation is selected """
@app.callback(
    Output('move1_id', 'children'),
    Output('move2_id', 'children'),
    Output('move3_id', 'children'),
    Output('move4_id', 'children'),
    Output('move5_id', 'children'),
    Output('move6_id', 'children'),
    Output('move7_id', 'children'),
    Output('move8_id', 'children'),
    Output('move9_id', 'children'),
    Output('move10_id', 'children'),
    Input('report_type', 'value'),
    Input('class_dropdown', 'value'),
    Input('variation_dropdown', 'value'), prevent_initial_call = True
    )

# Function to create most common moves for variations report
def find_top_moves(report, opening, value):
    
    if report == 'OPT2':

        all_moves = games[games['variation'] == value].iloc[:, [i for i in range(10,20)]]
        dfs = []
        for col in all_moves.columns:
            top_values = []
            top_values = all_moves[col].mode()
            dfs.append(pd.DataFrame({col: top_values}).reset_index(drop=True))
        top_moves = pd.DataFrame(pd.concat(dfs, axis=1))
            
        return [
                create_card(top_moves, 'Move1', 'move1', color='white', fontcolor='black'),
                create_card(top_moves, 'Move2', 'move2', color='black'),
                create_card(top_moves, 'Move3', 'move3', color='white', fontcolor='black'),
                create_card(top_moves, 'Move4', 'move4', color='black'),
                create_card(top_moves, 'Move5', 'move5', color='white', fontcolor='black'),
                create_card(top_moves, 'Move6', 'move6', color='black'),
                create_card(top_moves, 'Move7', 'move7', color='white', fontcolor='black'),
                create_card(top_moves, 'Move8', 'move8', color='black'),
                create_card(top_moves, 'Move9', 'move9', color='white', fontcolor='black'),
                create_card(top_moves, 'Move10', 'move10', color='black')
                ]
    
    if report == 'OPT1':
        
        all_moves = games[games['opening'] == opening].iloc[:, [i for i in range(10,20)]]
        dfs=[]
        for col in all_moves.columns:
            top_values = []
            top_values = all_moves[col].mode()
            dfs.append(pd.DataFrame({col: top_values}).reset_index(drop=True))
        top_moves = pd.DataFrame(pd.concat(dfs, axis=1))
        top_moves = pd.DataFrame(top_moves.iloc[0,:]).transpose()

        col_list = ['move3', 'move4', 'move5', 'move6', 'move7', 'move8', 'move9', 'move10']
        
        for col in col_list:
            top_moves[col] = " "
            
        return [
                create_card(top_moves, 'Move1', 'move1', color='white', fontcolor='black'),
                create_card(top_moves, 'Move2', 'move2', color='black'),
                create_card(top_moves, 'Move3', 'move3', color='white'),
                create_card(top_moves, 'Move4', 'move4', color='black', fontcolor='black'),
                create_card(top_moves, 'Move5', 'move5', color='white'),
                create_card(top_moves, 'Move6', 'move6', color='black', fontcolor='black'),
                create_card(top_moves, 'Move7', 'move7', color='white'),
                create_card(top_moves, 'Move8', 'move8', color='black', fontcolor='black'),
                create_card(top_moves, 'Move9', 'move9', color='white'),
                create_card(top_moves, 'Move10', 'move10', color='black', fontcolor='black')
                ]
        

""" Callback function that returns figures using the provided the chart type, opening classification and opening variation"""
@app.callback(
    Output('plot1', 'children'),
    Output('plot2', 'children'),
    Output('plot3', 'children'),
    Output('plot4', 'children'),
    Output('plot5', 'children')
    ,
    [Input('report_type', 'value'),
     Input('class_dropdown', 'value'),
     Input('variation_dropdown', 'value')],
    # Holding output state until user enters all the form information: report type and opening
    [State('plot1', 'children'), State('plot2', 'children'),
     State('plot3', 'children'), State('plot4', 'children'),
     State('plot4', 'children')
    ])
# Add computation to callback function and return graphs    
def get_graph(chart, opening, var, children1, children2, c3, c4, c5):
    
    if chart == 'OPT1':
    
        # Compute required information for creating opening graphs from data
        results_df, winner_df, filtered_df, scatter_df = compute_data_choice1(top_games, games, opening)
                
        cafe_colors =  ['rgb(146, 123, 21)', 'rgb(175, 51, 21)', 'rgb(206, 206, 40)', 'rgb(177, 180, 34)']
        
        results_fig = px.pie(results_df, names=results_df.index, values=results_df[opening],
                             title='Top10 Proportion of results', template='plotly_dark')
        results_fig.update_traces(marker=dict(colors=cafe_colors))
        
        winner_fig = px.pie(winner_df, names=winner_df.index, values=winner_df[opening],
                            title='Top10 Proportion of winner', template='plotly_dark')
        winner_fig.update_traces(marker=dict(colors=cafe_colors[0:3]))
        
        freq_fig = px.bar(filtered_df, x='variation', y='frequency', title='Top10 Frequency distribution by variation', 
                          height= 550, width=1235, color='frequency', text_auto=True, template='plotly_dark')
        freq_fig.update_layout(barmode='group', xaxis_tickangle=-45)
        
        scatter_fig = px.scatter(scatter_df, x='avg_turns', y='avg_opening_ply', size='frequency', size_max=40, hover_name=scatter_df.index, 
                                 title='Correlation between average opening ply and average number of turns', template='plotly_dark',
                                 hover_data=['avg_turns', 'avg_opening_ply'], color=scatter_df.index, labels=None).update_layout(showlegend=False)
        
        turns_fig = px.box(filtered_df, x='avg_turns', orientation='h',points='all', title='Top10 Average turns statistical distribution',
                           template='plotly_dark')
    
        return [
                dcc.Graph(figure=results_fig),
                dcc.Graph(figure=winner_fig),
                dcc.Graph(figure=scatter_fig),
                dcc.Graph(figure=freq_fig),
                dcc.Graph(figure=turns_fig)
                ]
    
    else:
    
        # Compute required information for creating variation graphs from data
        results_df, winner_df, hist_df, rated_df = compute_data_choice2(df=top_games, df2=games, value=var)
        
        results_fig = px.pie(results_df, names=results_df.index, values=results_df[var], 
                             title='Variation proportion of results', template='plotly_dark', hole=0.4,
                             color_discrete_sequence=px.colors.sequential.RdBu)
        
        winner_fig = px.pie(winner_df, names=winner_df.index, values=winner_df[var], hole=0.4,
                            title='Variation proportion of winner', template='plotly_dark',
                            color_discrete_sequence=px.colors.sequential.RdBu)
        
        hist_fig = px.histogram(hist_df, x=hist_df['turns'], color=hist_df['opening_ply'], template='plotly_dark',
                                title='Statistical distribution of number of turns for all games', pattern_shape=hist_df['rated'])
        
        rated_fig = px.pie(rated_df, names=rated_df.index, values=rated_df['count'], title='Percentage of rated games',
                           template='plotly_dark', hole=0.4,
                           color_discrete_sequence=px.colors.sequential.RdBu)
        
        grouped_box_fig = compute_grouped_boxplot(df=top_games, df2=games, value=opening)
        
        return[
                dcc.Graph(figure=results_fig),
                dcc.Graph(figure=winner_fig),
                dcc.Graph(figure=hist_fig),
                dcc.Graph(figure=grouped_box_fig),
                dcc.Graph(figure=rated_fig)
                ]
    
if __name__ == '__main__':
    app.run_server()