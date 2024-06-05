from recognise_text import bp as recognise_text_bp


@recognise_text_bp.post('/recognise_text')
def index():
    return 'Hello'
