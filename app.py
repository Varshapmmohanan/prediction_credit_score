from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model
with open('random_forest_model(2).joblib','rb') as f:
  model=joblib.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define prediction endpoint
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    Age = float(request.form['Age'])
    Annual_Income = float(request.form['Annual_Income'])
    Num_of_Delayed_Payment = float(request.form['Num_of_Delayed_Payment'])
    Credit_Utilization_Ratio = float(request.form['Credit_Utilization_Ratio'])
    Outstanding_Debt = float(request.form['Outstanding_Debt'])
    
    # Create DataFrame from input data
    input_data = pd.DataFrame({'Age': [Age], 
                               'Annual_Income': [Annual_Income], 
                               'Num_of_Delayed_Payment': [Num_of_Delayed_Payment], 
                               'Credit_Utilization_Ratio': [Credit_Utilization_Ratio], 
                               'Outstanding_Debt': [Outstanding_Debt]})
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Render results template with prediction
    return render_template('results.html', prediction=prediction)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)