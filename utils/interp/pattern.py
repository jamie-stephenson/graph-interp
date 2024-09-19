"""Helper functions for interpretting attention patterns"""

import plotly.graph_objs as go
from dash import dcc, html, Input, Output, Dash
import networkx as nx
from torch import Tensor
from jaxtyping import Float

import colorsys

def plot_attention(graph: nx.Graph, attention: Float[Tensor, "n_heads n_vertices n_vertices"]):

    # TODO: cannot run twice in a notebook with different attention patterns with out interference, fix this.
    if len(attention.shape)==4 and attention.shape[0]==1:
        attention = attention.squeeze(0)
    
    n = len(graph.nodes())
    n_heads = len(attention)

    assert n == attention.shape[-1] == attention.shape[-2],(
        f"Graph has {n} vertices but attention has shape {attention.shape}."
    )

    planar, cert = nx.check_planarity(graph)
    pos = nx.combinatorial_embedding_to_pos(cert) if planar else nx.spring_layout(graph)

    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=4, color='#888'),
        hoverinfo='none', mode='lines'
    )

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text',
        hoverinfo='text', marker=dict(
            showscale=False, color=[], size=40, line=dict(width=2)
        ),
        textposition="middle center",
        textfont=dict(size=16, color='black', family="Computer Modern, serif", weight="bold")
    )

    for node in graph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    node_trace.text = [str(i) for i in range(len(graph.nodes))]

    app = Dash(__name__)

    colorscales = get_colorscale(n)

    # Define the layout of the app
    app.layout = html.Div([
        html.Div(
            children=[
                html.Div(
                    dcc.Graph( 
                        figure=go.Figure(
                            data=go.Heatmap(
                                z=attention[i],
                                zmin=0,
                                zmax=1,
                                colorscale=colorscales[i],
                                showscale=False
                            ), 
                            layout=go.Layout(
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed')
                            )
                        ),
                        style={'width': '120px', 'height': '120px'}
                    ),
                    id=f'attn-{i}',
                ) 
                for i in range(n_heads)
            ],
            style={'display': 'flex', 'gap': '10px'}  # Flexbox layout with spacing between elements
        ),
        dcc.Graph(
            id='network-graph',
            figure=go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Interactive Graph Visualization',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))),
            style={'width': '1030px', 'height': '500px'}
        ),
    ])

    @app.callback(
        Output('network-graph', 'figure'),
        [Input(f'attn-{i}', 'n_clicks_timestamp') for i in range(n_heads)],
        Input('network-graph', 'hoverData')
    )
    def update_graph(*args):
        hoverData = args[-1]

        # Determine which matrix was clicked
        timestamps = args[:-1]

        # Determine which button was clicked most recently by finding the max timestamp
        if all(ts is None for ts in timestamps):
            button_index = 0  # Default to the first matrix if no button is pressed
        else:
            button_index = timestamps.index(max(ts for ts in timestamps if ts is not None))
        
        matrix = attention[button_index]

        colors = [0.5] * len(graph.nodes)

        if hoverData is not None:
            hovered_node = hoverData['points'][0]['pointIndex']
            if 0 <= hovered_node <= n:
                for j in range(len(graph.nodes)):
                    colors[j] = matrix[hovered_node, j].item()

        node_trace.marker.color = colors
        node_trace.marker.colorscale = colorscales[button_index]
        node_trace.marker.cmin = 0
        node_trace.marker.cmax = 1

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return fig

    app.run_server(debug=True)

def get_rgb_color(i, n):
    # Ensure i is within the correct range
    i = i % n

    hue = i / n  
    saturation = 1.0 
    value = 1.0  

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    r,g,b = tuple(int(c * 255) for c in rgb)

    return f'rgb({r},{g},{b})'

def get_colorscale(n):
    colorscales = []
    for i in range(n):
        color = get_rgb_color(i, n)
        colorscale = [
            [0, 'rgb(255,255,255)'],  # White
            [0.5, color],  # Specific color
            [1, 'rgb(0,0,0)']  # Black
        ]
        colorscales.append(colorscale)
    return colorscales
