import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import threading
import queue
import time
import numpy as np
import logging

# Suppress Werkzeug logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class TrainingVisualizer:
    def __init__(self):
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.DARKLY],
                           update_title=None)  # Remove "Updating..." from title
        self.data_queue = queue.Queue()
        self.node_stats = {}
        self.losses = []
        self.timestamps = []
        self.node_activities = {}
        self.network_stats = {}
        self.server = None
        self.is_running = False
        
        self.app.layout = self.create_layout()
        self.setup_callbacks()
    
    def create_layout(self):
        return dbc.Container([
            html.H1("Distributed Training Visualization", 
                   className="text-center my-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Loss"),
                        dbc.CardBody(dcc.Graph(id='loss-plot'))
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Node Activity"),
                        dbc.CardBody(dcc.Graph(id='node-activity'))
                    ])
                ], width=4)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Network Statistics"),
                        dbc.CardBody(dcc.Graph(id='network-stats'))
                    ])
                ])
            ], className="mt-4"),
            
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('loss-plot', 'figure'),
             Output('node-activity', 'figure'),
             Output('network-stats', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_graphs(_):
            # Process any new data
            while not self.data_queue.empty():
                data = self.data_queue.get()
                self.process_data(data)
            
            return (
                self.create_loss_plot(),
                self.create_node_activity_plot(),
                self.create_network_stats_plot()
            )
    
    def process_data(self, data):
        if 'loss' in data:
            self.losses.append(data['loss'])
            self.timestamps.append(time.time())
        
        if 'node_id' in data:
            node_id = data['node_id']
            if node_id not in self.node_activities:
                self.node_activities[node_id] = []
            self.node_activities[node_id].append(time.time())
        
        if 'network_stats' in data:
            self.network_stats = data['network_stats']
    
    def create_loss_plot(self):
        return {
            'data': [go.Scatter(
                y=self.losses,
                mode='lines',
                name='Training Loss',
                line=dict(color='#00ff00', width=2)
            )],
            'layout': go.Layout(
                title='Training Loss Over Time',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            )
        }
    
    def create_node_activity_plot(self):
        data = []
        for node_id, timestamps in self.node_activities.items():
            recent_activity = [t for t in timestamps if t > time.time() - 5]
            activity_level = len(recent_activity)
            data.append(go.Bar(
                x=[f'Node {node_id}'],
                y=[activity_level],
                name=f'Node {node_id}'
            ))
        
        return {
            'data': data,
            'layout': go.Layout(
                title='Node Activity (Last 5s)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            )
        }
    
    def create_network_stats_plot(self):
        if not self.network_stats:
            return go.Figure()
        
        nodes = list(self.network_stats.keys())
        throughput = [self.network_stats[node]['throughput'] for node in nodes]
        compression = [self.network_stats[node]['compression'] for node in nodes]
        
        return {
            'data': [
                go.Bar(name='Throughput (MB/s)', 
                      x=[f'Node {n}' for n in nodes], 
                      y=throughput),
                go.Bar(name='Compression Ratio', 
                      x=[f'Node {n}' for n in nodes], 
                      y=compression)
            ],
            'layout': go.Layout(
                title='Network Performance',
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            )
        }
    
    def update_data(self, data):
        self.data_queue.put(data)
    
    def start(self, port=8050):
        if not self.is_running:
            self.is_running = True
            self.server_thread = threading.Thread(
                target=self._run_server, 
                args=(port,),
                daemon=True  # Make thread daemon so it exits when main program exits
            )
            self.server_thread.start()
            time.sleep(1)  # Give server time to start
    
    def _run_server(self, port):
        self.app.run_server(port=port, debug=False, use_reloader=False)
    
    def shutdown(self):
        """Cleanup visualization resources"""
        if self.is_running:
            self.is_running = False
            # Clear all data
            self.data_queue = queue.Queue()
            self.losses = []
            self.timestamps = []
            self.node_activities = {}
            self.network_stats = {}