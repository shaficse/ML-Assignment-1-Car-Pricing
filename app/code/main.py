# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
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
    html.Div([
        dcc.Markdown('''
        ## How Prediction Works
        
        - **To predict the selling price of a used car, please provide the required inputs**
        - **Click the "Predict" button. The app will then use a trained model to predict the selling price based on the provided inputs**
        - **If you leave any input blank, the missing input will be automatically filled using a training imputation technique**
        ''')
    ]),
    dbc.Row([
        html.Div([
            
            dbc.Label("Year of Car Made (eg. 2020)"),
            dbc.Input(id="year", type="number", placeholder="Enter the Car Model Year"),
            html.Br(),
            dbc.Label("Number of km Drived (eg. 450000 km)"),
            dbc.Input(id="km_driven", type="number", placeholder="Enter KM drived"),
            html.Br(),
            dbc.Label("Size of Engine(eg. 1248 CC)"),
            dbc.Input(id="engine_size", type="number", placeholder="Enter Engine size (in CC)"),
            html.Br(),
            dbc.Label("Type of Fuel"),
            dcc.Dropdown(['Petrol', 'Diesel'], id='fuel_dropdown'),
            html.Br(),
            dbc.Label("Type of Transmission"),
            dcc.Dropdown(['Manual', 'Automatic'], id='transmission_dropdown'),
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
    State( component_id="fuel_dropdown", component_property="value"),
    State( component_id="transmission_dropdown", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def Predict_Life_Expectancy(year, km_driven, engine_size, fuel, transmission, submit):
    print(year, km_driven, engine_size, fuel, transmission)
    if year is None:
        age = 7.137924897668625 #initialized by mean of age
    else:
        age = 2020+1 - year  #calculating age as the same way was done in training
    if km_driven is None:
        km_driven = 70029.87346502936 #initialized by mean of km_driven
    if engine_size is None:
        engine_size = 1463.855626715462 #initialized by mean of engine_size
    if fuel is None or fuel == "Diesel":
        fuel = 0 #initialized by Diesel type if no input
    else:
        fuel = 1
    if transmission is None  or transmission == "Manual": 
        transmission = 1            #initialized by Manual type if no input
    else:
        transmission = 0

    #type casting of value in float64
    age = np.float64(age)
    km_driven = np.float64(km_driven)
    engine_size = np.float64(engine_size)
    fuel = np.float64(fuel)
    transmission = np.float64(transmission)
    
    # Make prediction using the model
    input_feature = np.array([[km_driven, age, engine_size,fuel,transmission]])
    print(input_feature.shape)
    prediction = model.predict(input_feature)[0]
    prediction = np.exp(prediction)
    predictedText = f"Predicted Selling Price: {prediction:.2f}"
    return predictedText
# Run the app
if __name__ == '__main__':
    app.run(debug=True)