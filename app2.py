from flask import Flask, render_template, request, redirect, url_for
from predict import predict_hcc_risk

app = Flask(__name__)

@app.route('/')
def home():
    """Home page with the input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the sequence submission and return prediction"""
    sequence = request.form.get('sequence', '')
    
    if not sequence or len(sequence) < 100:
        return render_template('index.html', 
                             error="Please paste a valid HBV sequence",
                             result=None,
                             synergies=None)
    
    try:
        result = predict_hcc_risk(sequence)
        synergies = result.get('synergies', [])
        
        return render_template('index.html', 
                             result=result, 
                             synergies=synergies,
                             error=None)
    except Exception as e:
        return render_template('index.html', 
                             error=f"Error processing sequence: {str(e)}",
                             result=None,
                             synergies=None)

# This handles people who type /predict directly in their browser
@app.route('/predict', methods=['GET'])
def predict_get():
    """Redirect GET requests to home page"""
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)