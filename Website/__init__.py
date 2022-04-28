import os
from flask import Flask

def create_app():
	# Creating the flask app itself
	app = Flask(__name__)

	# Giving the website a secret key to protect all user
	# cookie data.
	app.config["SECRET_KEY"] = os.environ['s_key']

	# Import our blueprint variables from the views and auth files
	from .views import views

	# Registering the blueprints from views and auth
	app.register_blueprint(views, url_prefix='/')

	return app