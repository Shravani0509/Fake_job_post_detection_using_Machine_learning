@app.route('/predict', methods=['POST'])
@login_required
def predict():
    global total_posts_checked, fraudulent_posts, non_fraudulent_posts

    job_description = request.form['job_description']
    location = request.form['location']

    try:
        
        text_features = text_transformer.transform([job_description])

       
        try:
            location_encoded = location_encoder.transform([location])
        except ValueError:
            location_encoded = np.array([-1])  
            
            suggestions = [loc for loc in common_indian_locations
                           if location.lower() in loc.lower()]
        else:
            suggestions = None

       
        features = hstack([text_features, location_encoded.reshape(1, -1)])

       
        proba = model.predict_proba(features)
        threshold = 0.6
        result = 'Fraudulent' if proba[0][1] > threshold else 'Not Fraudulent'
        confidence = round(max(proba[0]) * 100, 1)

       
        total_posts_checked += 1
        if result == 'Fraudulent':
            fraudulent_posts += 1
        else:
            non_fraudulent_posts += 1

        
        if result == 'Fraudulent':
            explanation = "The job posting contains characteristics commonly found in fraudulent listings, such as suspicious keywords or unrealistic offers."
        else:
            explanation = "The job posting appears legitimate based on the analysis of the description and location."

        return render_template('index.html',
                               prediction=result,
                               confidence=confidence,
                               explanation=explanation,
                               location_suggestions=suggestions)

    except ValueError as e:
       
        return render_template('index.html',
                               prediction='Error',
                               error_message=str(e))
