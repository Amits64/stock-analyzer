from flask import Flask, render_template
from dash_app import create_dash_app

app = Flask(__name__)

# Initialize the Dash app
dash_app = create_dash_app(app)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
