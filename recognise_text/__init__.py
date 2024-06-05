from flask import Blueprint

bp = Blueprint('recognise_text', __name__)

from recognise_text import routes
