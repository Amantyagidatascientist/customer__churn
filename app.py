from flask import Flask, render_template,request
import pandas as pd
import pickle


with open('./artifacts/transform_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('./artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__, template_folder='./tamplates')

@app.route('/')
def home():
    return render_template("index.html")  

@app.route('/analysis')
def analysis():
    return render_template("Automated_EDA.html")



@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        try:
            age = int(request.form.get('age', 0))
            last_login = int(request.form.get('last_login', 0))
            avg_time_spent = float(request.form.get('avg_time_spent', 0.0))
            avg_transaction_value = float(request.form.get('avg_transaction_value', 0.0))
            last_visit_time = request.form.get('time', '')
            points_in_wallet = float(request.form.get('points_in_wallet', 0.0))
            date = request.form.get('date', '')
            gender = request.form.get('gender', 'Unknown')
            region_category = request.form.get('region_category', 'Unknown')
            membership_category = request.form.get('membership_category', 'Unknown')
            joined_through_referral = request.form.get('joined_through_referral', 'No')
            preferred_offer_types = request.form.get('preferred_offer_types', 'None')
            medium_of_operation = request.form.get('medium_of_operation', 'Unknown')
            internet_option = request.form.get('internet_option', 'Unknown')
            used_special_discount = request.form.get('used_special_discount', 'No')
            offer_application_preference = request.form.get('offer_application_preference', 'No')
            past_complaint = request.form.get('past_complaint', 'No')
            feedback = request.form.get('feedback', 'None')

            # Creating a DataFrame for the input data
            data = {
                'age': [age],
                'days_since_last_login': [last_login],
                'avg_time_spent': [avg_time_spent],
                'avg_transaction_value': [avg_transaction_value],
                'points_in_wallet': [points_in_wallet],
                'gender': [gender],
                'region_category': [region_category],
                'membership_category': [membership_category],
                'joining_date': [date],
                'joined_through_referral': [joined_through_referral],
                'preferred_offer_types': [preferred_offer_types],
                'medium_of_operation': [medium_of_operation],
                'internet_option': [internet_option],
                'last_visit_time': [last_visit_time],
                'used_special_discount': [used_special_discount],
                'offer_application_preference': [offer_application_preference],
                'past_complaint': [past_complaint],
                'feedback': [feedback]
            }

            data_df = pd.DataFrame(data)

            # Apply transformations
            transformed_data = pipeline.transform(data_df)

            # Predict using the model
            prediction = model.predict(transformed_data)[0]

            # Return the prediction result
            return render_template("prediction.html", prediction_text=f"Churn Score is {prediction}")
        except Exception as e:
            return render_template("prediction.html", error=f"Error during prediction: {e}")
    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)
