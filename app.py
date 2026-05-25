from flask import Flask, request, render_template
from predict import predict_hcc_risk

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence']

    if len(sequence.strip()) < 100:
        return render_template('index.html',
            error="Sequence is too short. Please paste a complete HBV genome sequence.")

    try:
        result, confidence, model_features, additional_features = predict_hcc_risk(sequence)
        return render_template('index.html',
            result=result,
            confidence=confidence,
            model_features=model_features,
            additional_features=additional_features)
    except Exception as e:
        return render_template('index.html',
            error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)