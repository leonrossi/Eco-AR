from flask import render_template
from app import app

# @app.route("/home")
# @app.route("/")
# def home():
#     return render_template('home.html')

@app.route("/")
def index():
    return render_template('index.html')