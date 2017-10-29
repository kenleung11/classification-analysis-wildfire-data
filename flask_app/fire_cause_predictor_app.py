import flask
import pickle
from sklearn.externals import joblib
import numpy as np

#---------- MODEL IN MEMORY ----------------#

PREDICTOR = joblib.load('randf_miss_flask.pkl')

causes = {1: 'Lightning', 2: 'Equipment Use', 3: 'Smoking', 4: 'Campfire', 5: 'Debris Burning',
          6: 'Railroad', 7: 'Arson', 8: 'Children', 9: 'Miscellaneous', 10: 'Fireworks',
          11: 'Powerline', 12: 'Structure'}


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage


@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("index.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model


@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    prediction = PREDICTOR.predict_proba(x)[0]
    # Put the result in a nice dict so we can send it as json
    results = {"Lightning": prediction[0], "Equipment Use": prediction[1], "Smoking": prediction[2],
               "Campfire": prediction[3], "Debris Burning": prediction[4], "Railroad": prediction[5],
               "Arson": prediction[6], "Children": prediction[7], "Miscellaneous": prediction[8],
               "Fireworks": prediction[9], "Powerline": prediction[10], "Structure": prediction[11]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#


# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
