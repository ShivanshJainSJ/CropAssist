# Add these imports if needed
from flask import request, redirect, url_for, flash, session
from bson.objectid import ObjectId

# Add this new route to app.py just before the if __name__ == "__main__" line
@app.route('/batch_delete_detections', methods=['POST'])
@login_required
def batch_delete_detections():
    if not is_logged_in():
        flash("❌ Please log in to manage your disease detection history.")
        return redirect(url_for('login'))
    
    try:
        # Get the list of IDs to delete
        detection_ids = request.form.get('detection_ids', '').split(',')
        if not detection_ids or detection_ids[0] == '':
            flash("❌ No records selected for deletion.")
            return redirect(url_for('disease_history'))
        
        # Convert string IDs to ObjectId and verify user ownership
        obj_ids = []
        for id_str in detection_ids:
            obj_id = ObjectId(id_str.strip())
            detection = disease_history_collection.find_one({"_id": obj_id})
            
            if detection and detection.get("user_id") == session['user_id']:
                obj_ids.append(obj_id)
        
        # Delete the detections
        if obj_ids:
            result = disease_history_collection.delete_many({"_id": {"$in": obj_ids}})
            if result.deleted_count > 0:
                flash(f"✅ Successfully deleted {result.deleted_count} detection records.")
            else:
                flash("❌ No records were deleted.")
        else:
            flash("❌ You don't have permission to delete these records.")
            
    except Exception as e:
        flash(f"❌ Error deleting detections: {str(e)}")
    
    return redirect(url_for('disease_history'))

# Modified disease_history route with date filtering
@app.route('/disease_history')
@login_required
def disease_history():
    if not is_logged_in():
        flash("❌ Please log in to view your disease detection history.")
        return redirect(url_for('login'))
    
    # Get filter parameters
    plant_filter = request.args.get('plant', 'all')
    health_filter = request.args.get('health', 'all')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Build the query filter
    query = {"user_id": session['user_id']}
    
    # Add plant type filter
    if plant_filter != 'all':
        query["disease"] = {"$regex": f"^{plant_filter}___"}
    
    # Add health status filter
    if health_filter == 'healthy':
        if "disease" in query:
            query["disease"] = {"$and": [query["disease"], {"$regex": "healthy$"}]}
        else:
            query["disease"] = {"$regex": "healthy$"}
    elif health_filter == 'diseased':
        if "disease" in query:
            query["disease"] = {"$and": [query["disease"], {"$regex": "^((?!healthy).)*$"}]}
        else:
            query["disease"] = {"$regex": "^((?!healthy).)*$"}
            
    # Add date range filter
    if date_from:
        if date_to:
            query["timestamp"] = {"$gte": date_from, "$lte": date_to + " 23:59:59"}
        else:
            query["timestamp"] = {"$gte": date_from}
    elif date_to:
        query["timestamp"] = {"$lte": date_to + " 23:59:59"}
    
    # Get the user's detection history
    user_history = list(disease_history_collection.find(query).sort("timestamp", -1))
    
    # Get unique plant types from the user's history for the filter dropdown
    all_detections = list(disease_history_collection.find({"user_id": session['user_id']}))
    plant_types = set()
    for detection in all_detections:
        disease = detection.get('disease', '')
        if '___' in disease:
            plant_type = disease.split('___')[0]
            plant_types.add(plant_type)
    
    # Calculate some statistics
    total_detections = len(all_detections)
    healthy_count = sum(1 for d in all_detections if 'healthy' in d.get('disease', ''))
    disease_count = total_detections - healthy_count
    health_percentage = int((healthy_count / total_detections) * 100) if total_detections > 0 else 0
    
    # Get the most recent detection date
    most_recent_date = all_detections[0].get('timestamp') if all_detections else None
    
    return render_template('disease_history.html', 
                           history=user_history, 
                           user_name=session.get('name', 'User'),
                           plant_types=sorted(list(plant_types)),
                           current_plant=plant_filter,
                           current_health=health_filter,
                           date_from=date_from,
                           date_to=date_to,
                           stats={
                               'total': total_detections,
                               'healthy': healthy_count,
                               'diseased': disease_count,
                               'health_percentage': health_percentage,
                               'most_recent': most_recent_date
                           })
