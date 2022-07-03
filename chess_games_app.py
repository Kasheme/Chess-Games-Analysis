import dash
import dash_bootstrap_components as dbc

# create the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}])

# Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True

server = app.server