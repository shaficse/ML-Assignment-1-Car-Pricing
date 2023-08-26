# Import packages
from dash import Dash, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import pickle
import numpy as np


#Load Model
model_path = "model/car_prediction_MSE_0.072.model"
model = pickle.load(open(model_path, 'rb'))

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            dbc.Label("Year"),
            dbc.Input(id="year", type="number", placeholder="Enter the Car Manufacturing Year"),
            html.Br(),
            dbc.Label("KM Drived"),
            dbc.Input(id="km_driven", type="number", placeholder="Enter KM drived"),
            html.Br(),
            dbc.Label("Engine size (CC)"),
            dbc.Input(id="engine_size", type="number", placeholder="Enter Engine size (CC)"),
            html.Br(),
            dbc.Label("Fuel Type:"),
            dbc.Input(id="fuel", type="number", placeholder="Enter Fuel Type 1 for petrol, 0 for diesel"),
            html.Br(),
            dbc.Label("Transmission Type:"),
            dbc.Input(id="transmission", type="number", placeholder="Enter Fuel Type 1 for Manual, 0 for Automatic"),
            html.Br(),
            dbc.Button(id="submit", children="Predict", color="primary"),
        ],
        className="input_object"),

        html.Div(
            [
                # html.Br(),
                # dbc.Label("Predicted Life Expectancy is: "),
                html.Output(id="ouput_monitor", children="")
            ],
            className="output_object")
    ])

], fluid=True)

@callback(
    Output(component_id="ouput_monitor", component_property="children"),
    State(component_id="year", component_property="value"),
    State(component_id="km_driven", component_property="value"),
    State(component_id="engine_size", component_property="value"),
    State(component_id="fuel", component_property="value"),
    State(component_id="transmission", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def Predict_Life_Expectancy(year, km_driven, engine_size, fuel, transmission, submit):

    if year is None:
        age = 7.137924897668625
    else:
        age = 2020+1 - year
    if km_driven is None:
        km_driven = 70029.87346502936
    if engine_size is None:
        engine_size = 1463.855626715462
    if fuel is None:
        fuel = 1
    if transmission is None:
        transmission = 1

    try:
        age = np.float64(age)
        km_driven = np.float64(km_driven)
        engine_size = np.float64(engine_size)
        fuel = np.float64(fuel)
        transmission = np.float64(transmission)
    except ValueError:
        return "Please Enter Valid Numeric Values"
    
    # Make prediction using the model
    prediction = model.predict(np.array([[km_driven, age, engine_size,fuel,transmission]]) )[0]
    prediction = np.exp(prediction)
    return f"Predicted Selling-Price: {prediction:.2f}"
# Run the app
if __name__ == '__main__':
    app.run(debug=True)