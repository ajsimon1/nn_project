import os
from app.app import app

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hey food palace with bears'
    UPLOAD_FOLDER = os.path.join(app.instance_path, '\images')
