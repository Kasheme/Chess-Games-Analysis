from pydoc import classname
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

"""
Markdown text
"""

about_text = dcc.Markdown(
    """
    The aim of this project was to analyse 20,000 games of chess.  The overarching objective was to uncover common patterns and trends related to chess openings. 
    Primarily, I sought out to reveal the most popular chess openings and opening variations, and inform a Beginner-Intermediate player of popular and successful opening strategies, based off the findings.

    You have two options when selecting a report: Openings report & Variations report

    > ##### Openings report
    > Select an opening classification from the dropdown list of the top10 most used opening classifications in the original dataset.
    > Keep in mind, an opening classification only entails 1-2 moves of a chess game. Hence only the first 2 most common moves will be displayed on the moves board.
    > An example of a chess opening classification is: **Sicillian defense**

    > ##### Variations report
    > Once you've selected a top10 opening classification, select an opening variation in order to analyse the performance of the variation in the original dataset.
    > The average opening ply (the number of moves in the opening phase of a game) is between 6-10 moves. Therefore, the first 10 most common moves will be displayed on the moves board.
    > An example of an opening variation is: **Sicillian defense: Bowdler Attack**

    It is advisable that you have a chess board in front of you so you can interpret the various moves that are provided on the moves board for each opening and variation. This will help with your understanding and learning.
    """
)

defintions_text = dcc.Markdown(
    """
    Find below some definitions to help you with navigating the dashboard:
    * **Opening ply** - the number of moves in the opening phase of a game
    * **Turns** - the total number of turns between player white and player back in a game
    * **Frequency** - the number of occurences in the original games or top games dataset for a variation or opening
    * **Rated** - if the game is rated, then it is counted as an official game
    * **Mate** - short for *Checkmate*, mate is a game position in chess where the opponent's King is threatened with capture and there is no possible escape
    * **Resign** - a voluntary way of accepting defeat in a game
    * **Out of time** - if the game is timed, the eventual loser's allocated time has depleted before the winner's time has
    """)

about_card = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("An Introduction to the Project"),
                        html.P(about_text)
                    ],
                    className = "mt-4"
))
    
definitions_card = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("Key Definitions"),
                        html.P(defintions_text),
        
                    ],
                    className = "mt-4"
))

# ===== Build tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(about_card, tab_id="tab1", label="About"),
        dbc.Tab(definitions_card, tab_id="tab2", label="Key Definitions")

    ],
    id="tabs",
    active_tab="tab1",
    className="pb-4"
    )

# Connect to main app.py file
from chess_games_app import app
from chess_games_app import server

# Connect to app pages
from apps import chess_games_about

# to change later\
layout = dbc.Container(
    [
    dbc.Row(
        [
            dbc.Col(tabs, width=12, lg=7, className="mt-4 border")
        ],
        className="ms-1"
    )
# layout ends
    ],
    fluid=True
)

if __name__ == "__main__":
    app.run_server()