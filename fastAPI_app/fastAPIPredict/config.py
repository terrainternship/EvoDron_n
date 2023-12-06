# set  CORS_ORIGINS to your client URL
import os
class Config:
    SECRET_KEY = "justsomerandomstringstrictlyfordevelopment"
    CORS_ORIGINS = [
        "YOUR_CLIENT_URL"]
    DEFAULT_MODEL_NAME = 'best(ds 1.7)'
    DEFAULT_MODEL_PATH = os.path.join('/disk', 'models', DEFAULT_MODEL_NAME)