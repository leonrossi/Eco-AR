import os
from flask import render_template, request
from werkzeug.utils import secure_filename
from app import app
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from app.models import data_functions as func

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = '../files/'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
eventinfo = None

@app.route("/search", methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        form = request.form
        if 'guide' not in request.files:
            print("File don't found...")
        else:
            f = request.files['guide']
            k = form['kneig']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            filename = secure_filename(f.filename)
            f.seek(0)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        # Armazenando na arvore, se for a primeira busca
        global eventinfo
        if(eventinfo.data_struct is None):
            eventinfo = func.index_data(eventinfo)

        # Busca na arvore com elemento guia
        eventinfo, retrieval_components = func.knn_search(eventinfo, os.path.join(app.config['UPLOAD_FOLDER'],filename), k)
    
        # Visualizacao
        if(eventinfo.isCategorized):
            fig = go.Figure(data=px.scatter(eventinfo.components, x=0, y=1, opacity=0.3, color=eventinfo.df['label'])) # Build figure
        else:
            fig = go.Figure(data=px.scatter(eventinfo.components, x=0, y=1, opacity=0.3))
        # Add first scatter trace with medium sized markers
        # fig.add_trace(
        #     go.Scatter(
        #         x=retrieval_components[0,0],
        #         y=retrieval_components[0,1],
        #         opacity=1,
        #         name='retrieval',
        #         mode='markers',
        #         marker=dict(
        #             color=None,
        #             line=dict(
        #                 color='Red',
        #                 width=2
        #             )
        #         ),
        #         showlegend=True
        #     )
        # )

        fig.add_trace(
            go.Scatter(
                x=retrieval_components[:,0],
                y=retrieval_components[:,1],
                opacity=1,
                name='retrieval',
                mode='markers',
                marker=dict(
                    color=None,
                    line=dict(
                        color='Black',
                        width=2
                    )
                ),
                showlegend=True
            )
        )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)        
        
    return render_template('dist/index.html', graphJSON=graphJSON)


@app.route("/view", methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        form = request.form
        if('datafile' not in request.files):
            print("File don't found...")
        else:
            f = request.files['datafile']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            filename = secure_filename(f.filename)
            f.seek(0)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        # Recebendo dados do data frame
        global eventinfo
        eventinfo = func.get_data_information(os.path.join(app.config['UPLOAD_FOLDER'],filename))

        # Gerando os valores das projecoes em 2d
        eventinfo = func.get_data_projection(eventinfo, PCAflag = False, TSNEflag = True)

        # Visualizacao
        if(eventinfo.isCategorized):
            fig = go.Figure(data=px.scatter(eventinfo.components, x=0, y=1, color=eventinfo.df['label'])) # Build figure
        else:
            fig = go.Figure(data=px.scatter(eventinfo.components, x=0, y=1))

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)        
        
        return render_template('dist/index.html', graphJSON=graphJSON)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('dist/home.html')