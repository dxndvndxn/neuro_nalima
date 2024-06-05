from flask import Flask
from recognise_text import bp as recognise_text_bp


def create_app():
    app = Flask(__name__)

    app.register_blueprint(recognise_text_bp)

    return app
