# Updated fertility function with zero value validation
@app.route('/fertility', methods=['GET', 'POST'])
@login_required
def fertility():
    states = list(states_and_cities.keys())
    if request.method == 'POST':
        try:
            nitrogen = float(request.form['nitrogen'])
            phosphorous = float(request.form['phosphorous'])
            potassium = float(request.form['pottasium'])
            crop = request.form['crop'].strip().lower()
            
            # Validate that all nutrient values are greater than zero
            if nitrogen <= 0 or phosphorous <= 0 or potassium <= 0:
                flash("❌ Nutrient values must be greater than zero.")
                return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                   fertility_result="❌ All nutrient values (Nitrogen, Phosphorous, Potassium) must be greater than zero.")

            ideal_df = pd.read_csv('FertilizerData1.csv')
            fertilizer_df = pd.read_csv('fertilizer_composition.csv')

            if crop not in ideal_df['Crop'].str.lower().values:
                return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                   fertility_result=f"❌ Crop '{crop}' not found in ideal data.")

            ideal_values = ideal_df[ideal_df['Crop'].str.lower() == crop].iloc[0]
            ideal_N = ideal_values['N']
            ideal_P = ideal_values['P']
            ideal_K = ideal_values['K']

            deficiency_N = max(ideal_N - nitrogen, 0)
            deficiency_P = max(ideal_P - phosphorous, 0)
            deficiency_K = max(ideal_K - potassium, 0)

            N_content = fertilizer_df['N_content'].values
            P_content = fertilizer_df['P_content'].values
            K_content = fertilizer_df['K_content'].values

            c = np.ones(len(fertilizer_df))

            A = [
                -N_content,
                -P_content,
                -K_content
            ]
            b = [
                -deficiency_N,
                -deficiency_P,
                -deficiency_K
            ]

            bounds = [(0, None) for _ in range(len(fertilizer_df))]

            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

            if res.success:
                fertilizer_quantities = res.x
                recommendations = []
                for i, qty in enumerate(fertilizer_quantities):
                    if qty > 0:
                        fertilizer_name = fertilizer_df['Fertilizer'].iloc[i]
                        recommendations.append(f"Apply {qty:.2f} kg/ha of {fertilizer_name}")
                result = " | ".join(recommendations)
            else:
                result = "❌ No feasible fertilizer combination found."

            return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                               fertility_result=result)
                               
        except ValueError as e:
            # Handle input conversion errors
            error_message = f"❌ Invalid input values: {str(e)}"
            flash(error_message)
            return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                  fertility_result=error_message)
        except Exception as e:
            # Handle other unexpected errors
            error_message = f"❌ An error occurred: {str(e)}"
            flash(error_message)
            return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                  fertility_result=error_message)

    return render_template('fertility.html', states=states, states_and_cities=states_and_cities)
