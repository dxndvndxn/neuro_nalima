from flask import Flask
from ner import bp as ner_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(ner_bp)

    return app
