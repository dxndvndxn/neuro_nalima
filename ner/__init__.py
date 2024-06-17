from flask import Blueprint

bp = Blueprint('ner', __name__)

from ner import routes
