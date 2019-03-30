from flask import Flask, request, json, render_template
import logging
import scipy.stats as stat
import pandas as pd
from utils import *
from model import Model
import tensorflow as tf 


app = Flask(__name__, template_folder="view")
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

model = Model()
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
graph = tf.get_default_graph()
rating_df = pd.read_csv('ratings.csv',index_col='Filename')
s3 = boto3.resource('s3')

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/alive', methods=['GET'])
def alive():
    return "I am Alive"
    
@app.route('/predict', methods=['POST'])
def predict():

    f = request.files['img_file']
    f.save('./query_img.jpg')
    img = cv2.imread('./query_img.jpg')
    faces_coords = detect_face(img, face_classifier)

    if faces_coords == ():
        logger.info("no face found")
        res = { "error": "no face found" }    
        return json.dumps(res)
    if len(faces_coords) > 1:
        logger.info("multiple faces found")
        res = { "error": "multiple faces found"}    
        return json.dumps(res)

    face_crop = face_close_crop(img, faces_coords)
    object_url = upload_crop_to_s3(s3)
    img_array = img_preparation(face_crop)

    with graph.as_default():
        score = model.predict(img_array)[0][0]
        converted_score = round(stat.percentileofscore(rating_df['Rating'],score) / 10.0 , 1)
        res = { "score": str(converted_score) , "input_thumbnail": object_url }
    return json.dumps(res)

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8080)