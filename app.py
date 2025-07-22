from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            department = request.form['department']
            day = request.form['day']

            # Encoding categorical variables manually
            department_map = {
                'sweing': 0, 'finishing': 1
            }
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                'Thursday': 3, 'Friday': 4, 'Saturday': 5
            }

            dept_val = department_map.get(department.lower(), 0)
            day_val = day_map.get(day, 0)

            # Fetch and convert form inputs
            data = [
                float(request.form['wip']),
                float(request.form['over_time']),
                float(request.form['incentive']),
                float(request.form['idle_time']),
                float(request.form['idle_men']),
                float(request.form['no_of_style_change']),
                float(request.form['smv']),
                float(request.form['actual_productivity']),
                dept_val,
                day_val,
                float(request.form['line_number'])
            ]

            prediction = model.predict([np.array(data)])
            return render_template('result.html', prediction=round(prediction[0], 2))

        except Exception as e:
            return f"<h3>Error: {str(e)}</h3><a href='/predict'>Back</a>"

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
