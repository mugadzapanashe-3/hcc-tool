from flask import Flask, render_template, request, redirect, url_for
from predict import predict_hcc_risk

# Create the Flask app instance
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
                             error="Please paste a valid HBV sequence (at least 100 nucleotides)",
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

@app.route('/predict', methods=['GET'])
def predict_get():
    """Redirect GET requests to home page"""
    return redirect(url_for('home'))

# For Hugging Face - must use port 7860
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
