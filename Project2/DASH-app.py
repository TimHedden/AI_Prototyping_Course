import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.dependencies import ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import cohere




# Load the dataset
file_path = '~/food_data2_cleaned.csv'
data = pd.read_csv(file_path)

# Initialize Cohere client
cohere_api_key = 'cDGxDRAzWz97x4FbeLeoeq8go5Vk737qSbEhCA2p'
cohere_client = cohere.Client(cohere_api_key)


# Select relevant columns for the app
selected_columns = [
    'product_name', 'energy-kj_100g', 'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 
    'salt_100g', 'sodium_100g', 'nutriscore_grade', 'category', 'country'
]

data_selected = data[selected_columns].copy()

# Convert 'nutriscore_grade' to a numerical format for model training
nutriscore_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
nutriscore_reverse_mapping = {v: k for k, v in nutriscore_mapping.items()}
data_selected['nutriscore_grade_encoded'] = data_selected['nutriscore_grade'].map(nutriscore_mapping)

# Load the model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['/assets/DASH_Assignment2.css'])
app.title = "Food Nutrition Dashboard"

# Layout
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("Open_Food_Facts_logo_2022.png"))]
        ),
        # Sidebar and Content
        html.Div(
            id="sidebar-and-content",
            className="row",
            children=[
                # Sidebar
                html.Div(
                    id="sidebar",
                    className="four columns sidebar",
                    children=[
                        html.H2("Global Filters", className="sidebar-title"),
                        html.P("Select Country"),
                        dcc.Dropdown(
                            id="country-filter",
                            options=[{"label": country, "value": country} for country in data_selected['country'].unique()],
                            value=[country for country in data_selected['country'].unique()],  # Set to all countries
                            multi=True
                        ),
                        html.P("Select Nutrition Score"),
                        dcc.Dropdown(
                            id="nutrition-score-filter",
                            options=[{"label": f"{grade} ({score})", "value": score} for grade, score in nutriscore_mapping.items()],
                            value=[score for grade, score in nutriscore_mapping.items()],  # Set to all scores
                            multi=True
                        ),
                        html.P("Select Category"),
                        dcc.Dropdown(  # This line was missing
                            id="category-filter",
                            options=[{"label": category, "value": category} for category in data_selected['category'].unique()],
                            value=[category for category in data_selected['category'].unique()],  # Set to all categories
                            multi=True
                        ),
                        html.Button(
                            "Apply Filters", 
                            id="apply-filters", 
                            n_clicks=0, 
                            style={
                                'backgroundColor': '#007bff', 
                                'color': 'white', 
                                'border': 'none', 
                                'padding': '10px 20px', 
                                'fontSize': '16px', 
                                'cursor': 'pointer', 
                                'borderRadius': '4px', 
                                'width': '100%', 
                                'marginTop': '10px'
                            }
                        )
                    ]

                ),
                # Main content
                html.Div(
                    id="content",
                    className="eight columns content",
                    children=[
                        dcc.Tabs(
                            className="custom-tabs",
                            children=[
                                dcc.Tab(label='Data Dashboard', children=[
                                    html.Div(
                                        className="graph-container",
                                        children=[
                                            html.H4("Average Nutrition Metrics by Category", className="graph-title"),
                                            dcc.Graph(id='heatmap')
                                        ]
                                    ),
                                    html.Div(
                                        className="graph-container",
                                        children=[
                                            html.H4("Nutrition Metric Distribution (Boxplot)", className="graph-title"),
                                            html.Div(
                                                style={'display': 'flex', 'justify-content': 'space-between'},
                                                children=[
                                                    html.Div(
                                                        style={'width': '48%'},
                                                        children=[
                                                            dcc.Graph(id='boxplot-1'),
                                                            html.Label('Select Nutritional Metric:'),
                                                            dcc.Dropdown(
                                                                id='boxplot-metric-1',
                                                                options=[{'label': col, 'value': col} for col in selected_columns if col not in ['nutriscore_grade', 'product_name']],
                                                                value='energy-kcal_100g'
                                                            ),
                                                            html.Label('Select Country'),
                                                            dcc.Dropdown(
                                                                id="boxplot-country-filter-1",
                                                                options=[{"label": country, "value": country} for country in data_selected['country'].unique()],
                                                                value=[country for country in data_selected['country'].unique()],  # Set to all countries
                                                                multi=True
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        style={'width': '48%'},
                                                        children=[
                                                            dcc.Graph(id='boxplot-2'),
                                                            html.Label('Select Nutritional Metric:'),
                                                            dcc.Dropdown(
                                                                id='boxplot-metric-2',
                                                                options=[{'label': col, 'value': col} for col in selected_columns if col not in ['nutriscore_grade', 'product_name']],
                                                                value='fat_100g'
                                                            ),
                                                            html.Label('Select Country'),
                                                            dcc.Dropdown(
                                                                id="boxplot-country-filter-2",
                                                                options=[{"label": country, "value": country} for country in data_selected['country'].unique()],
                                                                value=[country for country in data_selected['country'].unique()],  # Set to all countries
                                                                multi=True
                                                            ),
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="graph-container",
                                        children=[
                                            html.H4("Nutrition Metric Distribution (Scatterplot)", className="graph-title"),
                                            dcc.Graph(id='scatter-plot'),
                                            html.Div(
                                                className="row",
                                                children=[
                                                    html.Div(
                                                        className="six columns",
                                                        children=[
                                                            html.P("X-axis:"),
                                                            dcc.Dropdown(
                                                                id='x-axis',
                                                                options=[{'label': col, 'value': col} for col in selected_columns if col not in ['nutriscore_grade', 'product_name']],
                                                                value='energy-kcal_100g'
                                                            )
                                                        ]
                                                    ),
                                                    html.Div(
                                                        className="six columns",
                                                        children=[
                                                            html.P("Y-axis:"),
                                                            dcc.Dropdown(
                                                                id='y-axis',
                                                                options=[{'label': col, 'value': col} for col in selected_columns if col not in ['nutriscore_grade', 'product_name']],
                                                                value='fat_100g'
                                                            )
                                                        ]
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    dcc.Checklist(
                                                        id='trendline-toggle',
                                                        options=[{'label': 'Do not show trendlines', 'value': 'show_trendlines'}],
                                                        value=[]
                                                    )
                                                ],
                                                style={'marginTop': '10px'}
                                            )
                                        ]
                                    )
,
                                    html.Div(
                                        className="graph-container",
                                        children=[
                                            html.H4("Nutrition correlation heatmap", className="graph-title"),
                                            dcc.Graph(id='nutrition-correlation-heatmap')
                                        ]
                                    )
                                ]),

                              dcc.Tab(label='Product recommender', children=[
                                html.Div(
                                    className="prediction-container",
                                    children=[
                                        html.H4('Hello there!'),
                                        html.P('Enter the nutrition metrics of your product and scroll down to find the respective Nutri score and a consumption recommendation.'),
                                        html.Div(
                                            className="row",
                                            style={'display': 'flex', 'flexWrap': 'wrap'},
                                            children=[
                                                html.Div(
                                                    className="six columns",
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Energy (kJ/100g)'),
                                                        dcc.Input(id='input-energy-kj', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                ),
                                                html.Div(
                                                    className="six columns",
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Energy (kcal/100g)'),
                                                        dcc.Input(id='input-energy-kcal', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="row",
                                            style={'display': 'flex', 'flexWrap': 'wrap'},
                                            children=[
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Fat (100g)'),
                                                        dcc.Input(id='input-fat', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                ),
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Saturated Fat (100g)'),
                                                        dcc.Input(id='input-saturated-fat', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="row",
                                            style={'display': 'flex', 'flexWrap': 'wrap'},
                                            children=[
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Carbohydrates (100g)'),
                                                        dcc.Input(id='input-carbohydrates', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                ),
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Sugars (100g)'),
                                                        dcc.Input(id='input-sugars', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="row",
                                            style={'display': 'flex', 'flexWrap': 'wrap'},
                                            children=[
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Fiber (100g)'),
                                                        dcc.Input(id='input-fiber', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                ),
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Proteins (100g)'),
                                                        dcc.Input(id='input-proteins', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="row",
                                            style={'display': 'flex', 'flexWrap': 'wrap'},
                                            children=[
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Salt (100g)'),
                                                        dcc.Input(id='input-salt', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                ),
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Sodium (100g)'),
                                                        dcc.Input(id='input-sodium', type='number', value=0, style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className="row",
                                            style={'display': 'flex', 'flexWrap': 'wrap'},
                                            children=[
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Category'),
                                                        dcc.Dropdown(
                                                            id='input-category',
                                                            options=[{'label': category, 'value': category} for category in data_selected['category'].unique()],
                                                            value=data_selected['category'].unique()[0],
                                                            style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}
                                                        )
                                                    ]
                                                ),
                                                html.Div(
                                                    className='six columns',
                                                    style={'flex': '1', 'padding': '10px', 'boxSizing': 'border-box'},
                                                    children=[
                                                        html.Label('Country'),
                                                        dcc.Dropdown(
                                                            id='input-country',
                                                            options=[{'label': country, 'value': country} for country in data_selected['country'].unique()],
                                                            value=data_selected['country'].unique()[0],
                                                            style={'width': '100%', 'padding': '8px', 'margin': '5px 0 10px 0', 'boxSizing': 'border-box', 'border': '1px solid #ccc', 'borderRadius': '4px'}
                                                        )
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Button(
                                            'Predict', 
                                            id='predict-button', 
                                            n_clicks=0, 
                                            style={
                                                'backgroundColor': '#007bff', 
                                                'color': 'white', 
                                                'border': 'none', 
                                                'padding': '10px 20px', 
                                                'fontSize': '16px', 
                                                'cursor': 'pointer', 
                                                'borderRadius': '4px', 
                                                'width': '100%', 
                                                'marginTop': '10px'
                                            }
                                        ),
                                        html.Div(id='prediction-output', style={'marginTop': '20px'}),
                                        html.Div(id='cohere-recommendation', style={'marginTop': '40px'}),
                                        html.Div(id='best-products', style={'marginTop': '40px'})
                                    ],
                                    style={'backgroundColor': 'white', 'padding': '20px'}
                                )


                            ])

                             






                            ],
                            style={'marginTop': '5.5rem'}  # Adjusted margin-top to push tabs below the banner
                        )
                    ]
                )
            ]
        )
    ],
)

# Define the callbacks for the app
@app.callback(
    Output('heatmap', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('country-filter', 'value'),
     State('nutrition-score-filter', 'value'),
     State('category-filter', 'value')]
)
def update_heatmap(n_clicks, selected_countries, selected_scores, selected_categories):
    filtered_df = data_selected.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_scores:
        filtered_df = filtered_df[filtered_df['nutriscore_grade_encoded'].isin(selected_scores)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    numeric_cols = ['energy-kj_100g', 'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 
                    'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 
                    'salt_100g', 'sodium_100g']
    avg_df = filtered_df.groupby('category')[numeric_cols].mean().reset_index()

    z = avg_df[numeric_cols].values
    x = numeric_cols
    y = avg_df['category']

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
        hoverongaps=False,
        zmin=0,
        zmax=np.max(z),
        text=z,
        texttemplate='%{text:.1f}',  # Display the values in the heatmap
        hovertemplate='Category: %{y}<br>Metric: %{x}<br>Average: %{z:.1f}<extra></extra>'
    ))

    fig.update_layout(
        title='',
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left'),
        margin=dict(l=70, b=50, t=50, r=50),
        height=400,
        hovermode='closest',
        showlegend=False
    )

    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('apply-filters', 'n_clicks'),
     Input('trendline-toggle', 'value')],
    [State('country-filter', 'value'),
     State('nutrition-score-filter', 'value'),
     State('category-filter', 'value')]
)
def update_scatter_plot(x_axis, y_axis, n_clicks, trendline_toggle, selected_countries, selected_scores, selected_categories):
    filtered_df = data_selected.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_scores:
        filtered_df = filtered_df[filtered_df['nutriscore_grade_encoded'].isin(selected_scores)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    fig = px.scatter(
        filtered_df, 
        x=x_axis, 
        y=y_axis, 
        color='category',  # Color by category
        hover_data=['product_name', 'nutriscore_grade'],
        labels={'category': 'Category'}
    )

    if 'show_trendlines' not in trendline_toggle:
        for category in filtered_df['category'].unique():
            category_df = filtered_df[filtered_df['category'] == category]
            trendline_fig = px.scatter(category_df, x=x_axis, y=y_axis, trendline='lowess', color_discrete_sequence=[fig.data[filtered_df['category'].unique().tolist().index(category)].marker.color])
            fig.add_traces(trendline_fig.data)
    
    fig.update_layout(
        legend=dict(
            x=1,
            y=1,
            traceorder='normal',
            title='Category'
        )
    )
    return fig




@app.callback(
    Output('nutrition-correlation-heatmap', 'figure'),
    Input('apply-filters', 'n_clicks'),
    [State('country-filter', 'value'),
     State('nutrition-score-filter', 'value'),
     State('category-filter', 'value')]
)
def update_nutrition_correlation_heatmap(n_clicks, selected_countries, selected_scores, selected_categories):
    filtered_df = data_selected.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_scores:
        filtered_df = filtered_df[filtered_df['nutriscore_grade_encoded'].isin(selected_scores)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    correlation_matrix = filtered_df.drop(['product_name', 'nutriscore_grade', 'nutriscore_grade_encoded', 'category', 'country'], axis=1).corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
    ))
    fig.update_layout(
        xaxis_nticks=36
    )
    return fig

@app.callback(
    [Output('boxplot-country-filter-1', 'options'),
     Output('boxplot-country-filter-2', 'options'),
     Output('boxplot-metric-1', 'options'),
     Output('boxplot-metric-2', 'options')],
    [Input('country-filter', 'value'),
     Input('category-filter', 'value')]
)
def update_boxplot_filters(selected_countries, selected_categories):
    # Filter the dataset based on the global filters
    filtered_df = data_selected.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Update country options based on global filters
    country_options = [{"label": country, "value": country} for country in filtered_df['country'].unique()]
    
    # Update category options based on global filters
    category_options = [{"label": category, "value": category} for category in filtered_df['category'].unique()]
    
    # Metric options remain the same as they are static
    metric_options = [{'label': col, 'value': col} for col in selected_columns if col not in ['nutriscore_grade', 'product_name']]
    
    return country_options, country_options, metric_options, metric_options

@app.callback(
    Output('boxplot-1', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('boxplot-metric-1', 'value'),
     State('boxplot-country-filter-1', 'value'),
     State('category-filter', 'value'),
     State('country-filter', 'value'),
     State('nutrition-score-filter', 'value')]
)
def update_boxplot_1(n_clicks, metric, custom_countries, selected_categories, selected_countries, selected_scores):
    filtered_df = data_selected.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_scores:
        filtered_df = filtered_df[filtered_df['nutriscore_grade_encoded'].isin(selected_scores)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    if custom_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(custom_countries)]

    fig = px.box(
        filtered_df,
        x='category',
        y=metric,
        color='category',
        boxmode="overlay",
        points='all',
        category_orders={"category": filtered_df['category'].unique()},
        hover_data={'product_name': True, 'nutriscore_grade': True}
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(showlegend=True, legend_title_text='Category')

    return fig


@app.callback(
    Output('boxplot-2', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('boxplot-metric-2', 'value'),
     State('boxplot-country-filter-2', 'value'),
     State('category-filter', 'value'),
     State('country-filter', 'value'),
     State('nutrition-score-filter', 'value')]
)
def update_boxplot_2(n_clicks, metric, custom_countries, selected_categories, selected_countries, selected_scores):
    filtered_df = data_selected.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    if selected_scores:
        filtered_df = filtered_df[filtered_df['nutriscore_grade_encoded'].isin(selected_scores)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    if custom_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(custom_countries)]

    fig = px.box(
        filtered_df,
        x='category',
        y=metric,
        color='category',
        boxmode="overlay",
        points='all',
        category_orders={"category": filtered_df['category'].unique()},
        hover_data={'product_name': True, 'nutriscore_grade': True}
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(showlegend=True, legend_title_text='Category')

    return fig


@app.callback(
    [Output('prediction-output', 'children'),
     Output('cohere-recommendation', 'children'),
     Output('best-products', 'children')],
    Input('predict-button', 'n_clicks'),
    [State('input-energy-kj', 'value'),
     State('input-energy-kcal', 'value'),
     State('input-fat', 'value'),
     State('input-saturated-fat', 'value'),
     State('input-carbohydrates', 'value'),
     State('input-sugars', 'value'),
     State('input-fiber', 'value'),
     State('input-proteins', 'value'),
     State('input-salt', 'value'),
     State('input-sodium', 'value'),
     State('input-category', 'value'),
     State('input-country', 'value')]
)
def predict_and_recommend(n_clicks, energy_kj, energy_kcal, fat, saturated_fat, carbohydrates, sugars, fiber, proteins, salt, sodium, selected_category, selected_country):
    if n_clicks > 0:
        input_data = [[energy_kj, energy_kcal, fat, saturated_fat, carbohydrates, sugars, fiber, proteins, salt, sodium]]
        input_data_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_data_scaled)
        grade = nutriscore_reverse_mapping[prediction[0]]
        prediction_result = f'Predicted Nutriscore Grade: {grade.upper()}'

        filtered_df = data_selected.copy()
        if selected_category:
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_country:
            filtered_df = filtered_df[filtered_df['country'] == selected_country]

        best_products = filtered_df[filtered_df['nutriscore_grade_encoded'] == 1]

        if best_products.empty:
            best_products_table = "No products found with the best Nutriscore."
        else:
            table_header = [
                html.Thead(html.Tr([html.Th(col) for col in ['nutriscore_grade', 'country', 'category', 'product_name', 'energy-kj_100g', 'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']]))
            ]
            table_body = [
                html.Tbody([
                    html.Tr([html.Td(best_products.iloc[i][col]) for col in ['nutriscore_grade', 'country', 'category', 'product_name', 'energy-kj_100g', 'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']])
                    for i in range(min(len(best_products), 10))
                ])
            ]
            best_products_table = table_header + table_body

        cohere_input = f"Energy: {energy_kj} kJ/100g, {energy_kcal} kcal/100g; Fat: {fat} g/100g; Saturated Fat: {saturated_fat} g/100g; Carbohydrates: {carbohydrates} g/100g; Sugars: {sugars} g/100g; Fiber: {fiber} g/100g; Proteins: {proteins} g/100g; Salt: {salt} g/100g; Sodium: {sodium} g/100g; Nutri-Score: {grade.upper()}"

        response = cohere_client.generate(
            model='command-xlarge-nightly',
            prompt=f"The following product has the following nutritional values: {cohere_input}. First tell the user, what the ingredients of the product mean and what the impact on its boddy is. In addition give some advise, how much of the of the product per serving or per day is recommended for the user to consume, in maximum to stay healthy. To give this recommendation rely on actual research figures and give some explicit numeric amounts.",
            max_tokens=250,
            temperature=0.7
        )
        cohere_recommendation = response.generations[0].text

        if grade.lower() != 'a':
            cohere_recommendation += ".\n\n Based in the predicted score you might want to consider one of the products in the table below for a healthier choice."

        return prediction_result, cohere_recommendation, html.Div([html.Table(best_products_table, style={'width': '100%', 'borderCollapse': 'collapse'})], style={'maxHeight': '300px', 'overflowY': 'auto'})

    return '', '', ''




# -- Main code to run the app --
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
