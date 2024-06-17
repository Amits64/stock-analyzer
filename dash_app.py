import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from data_fetcher import fetch_historical_data, preprocess_data, fetch_available_cryptocurrencies


def create_dash_app(flask_app):
    dash_app = dash.Dash(
        server=flask_app,
        name="DashApp",
        url_base_pathname='/dash/'
    )

    dash_app.layout = html.Div([
        html.H1("Cryptocurrency Dashboard"),
        dcc.Dropdown(
            id='crypto-dropdown',
            options=[{'label': coin, 'value': coin} for coin in fetch_available_cryptocurrencies()],
            value='bitcoin'
        ),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date='2020-01-01',
            end_date='2023-01-01'
        ),
        dcc.Graph(id='price-chart'),
        dcc.Graph(id='heatmap-chart'),
        dcc.Graph(id='trend-analysis-chart'),
        html.Div([
            dcc.Checklist(
                id='indicators-checklist',
                options=[
                    {'label': 'SMA', 'value': 'SMA'},
                    {'label': 'EMA', 'value': 'EMA'},
                    {'label': 'RSI', 'value': 'RSI'},
                    {'label': 'MACD', 'value': 'MACD'}
                ],
                value=['SMA', 'EMA']
            )
        ])
    ])

    @dash_app.callback(
        [Output('price-chart', 'figure'),
         Output('heatmap-chart', 'figure'),
         Output('trend-analysis-chart', 'figure')],
        [Input('crypto-dropdown', 'value'),
         Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('indicators-checklist', 'value')]
    )
    def update_graphs(selected_crypto, start_date, end_date, selected_indicators):
        df = fetch_historical_data(selected_crypto, start_date, end_date)
        df = preprocess_data(df)

        # Price Chart
        traces = [go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close')]
        if 'SMA' in selected_indicators:
            traces.append(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA'))
        if 'EMA' in selected_indicators:
            traces.append(go.Scatter(x=df.index, y=df['EMA_10'], mode='lines', name='EMA'))
        if 'RSI' in selected_indicators:
            traces.append(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis='y2'))
        if 'MACD' in selected_indicators:
            traces.append(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
            traces.append(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD Signal'))

        price_chart = {
            'data': traces,
            'layout': go.Layout(
                title='Price Chart',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                yaxis2={'title': 'RSI', 'overlaying': 'y', 'side': 'right'}
            )
        }

        # Heatmap
        heatmap_chart = {
            'data': [go.Heatmap(z=df.corr(), x=df.columns, y=df.columns)],
            'layout': go.Layout(title='Heatmap', xaxis={'title': 'Metrics'}, yaxis={'title': 'Metrics'})
        }

        # Trend Analysis
        trend_analysis_chart = {
            'data': [go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA')],
            'layout': go.Layout(title='Trend Analysis', xaxis={'title': 'Date'}, yaxis={'title': 'SMA'})
        }

        return price_chart, heatmap_chart, trend_analysis_chart

    return dash_app
