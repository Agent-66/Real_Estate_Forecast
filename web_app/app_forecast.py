from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
# load the model
model = pickle.load(open('cat_log_model.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    status = request.form['status']
    baths = int(request.form['baths'])
    fireplace = int(request.form['fireplace'])
    sqft = float(request.form['sqft'])
    zipcode = int(request.form['zipcode'])
    state = request.form['state']
    stories = int(request.form['stories'])
    year_built = int(request.form['year_built'])
    heating = int(request.form['heating'])
    cooling = int(request.form['cooling'])
    parking = int(request.form['parking'])
    property_add_type = request.form['property_add_type']
    private_pool = int(request.form['private_pool'])
    school_rating = float(request.form['school_rating'])
    school_min_distance = float(request.form['school_min_distance'])
    beds_area = float(request.form['beds_area'])
    baths_area = float(request.form['baths_area'])
    lotsize_sqft = float(request.form['lotsize_sqft'])
    
    result = np.exp(model.predict([[
        status, baths, fireplace, sqft, zipcode, state, stories, year_built, heating, cooling, parking, 
        property_add_type, private_pool, school_rating, school_min_distance, beds_area, baths_area, lotsize_sqft]])[0])
    
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)