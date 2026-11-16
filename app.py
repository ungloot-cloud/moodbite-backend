from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import psycopg2
import random
import math
from collections import Counter

app = Flask(__name__)
CORS(app) 

DATABASE_URL = os.environ.get("DATABASE_URL")

# ==========================================
# 1. K-MEANS CLUSTERING DATA (From Orange)
# ==========================================
FEATURE_MEANS = [2.97, 3.66, 3.64, 0.49, 0.31, 0.49, 0.43, 3.88]
FEATURE_STDS = [0.865506, 1.021959, 1.072567, 0.4999, 0.462493, 0.4999, 0.495076, 0.765245]

CLUSTER_CENTROIDS = {
    "C1": [0.034806, -0.14071, -0.20571, -0.33491, 1.49191, -0.33491, 0.239132, -0.306774],
    "C2": [0.805, 0.959923, 0.981231, 0.968908, -0.67028, 1.0202, -0.350629, 0.726718],
    "C3": [-1.082333, -1.1024, -1.063, -0.91352, -0.67028, -0.9802, 0.208725, -0.6272],
}

# ==========================================
# 2. DATABASE & CONSTANTS
# ==========================================
def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def init_db():
    if not DATABASE_URL: return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                dish_name TEXT,
                rating INTEGER,
                mood TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_tags (
                id SERIAL PRIMARY KEY,
                dish_name TEXT,
                tag TEXT,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_clusters (
                user_id TEXT PRIMARY KEY,
                cluster TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Init Error: {e}")

init_db()

SEED_DISHES = [
    {"id": 1, "name": "Dim Sum Platter", "emoji": "🥟", "veg": False, "type": "comfort"}, {"id": 2, "name": "Wonton Noodles", "emoji": "🍜", "veg": False, "type": "comfort"}, {"id": 3, "name": "Roast Goose Rice", "emoji": "🦆", "veg": False, "type": "comfort"}, {"id": 4, "name": "Pineapple Bun", "emoji": "🍞", "veg": True, "type": "reward"}, {"id": 5, "name": "Egg Tart", "emoji": "🥧", "veg": True, "type": "reward"}, {"id": 6, "name": "HK Milk Tea", "emoji": "🧋", "veg": True, "type": "reward"}, {"id": 7, "name": "Beef Brisket Noodles", "emoji": "🍜", "veg": False, "type": "comfort"}, {"id": 8, "name": "Claypot Rice", "emoji": "🍚", "veg": False, "type": "comfort"}, {"id": 9, "name": "Cheung Fun", "emoji": "🥢", "veg": True, "type": "comfort"}, {"id": 10, "name": "Curry Fish Balls", "emoji": "🍢", "veg": False, "type": "adventurous"}, {"id": 11, "name": "Siu Mei Platter", "emoji": "🍖", "veg": False, "type": "comfort"}, {"id": 12, "name": "HK French Toast", "emoji": "🍞", "veg": True, "type": "reward"}, {"id": 13, "name": "Red Bean Soup", "emoji": "🍲", "veg": True, "type": "reward"}, {"id": 14, "name": "Mango Pomelo Sago", "emoji": "🥭", "veg": True, "type": "reward"}, {"id": 15, "name": "Sushi Platter", "emoji": "🍣", "veg": False, "type": "healthy"}, {"id": 16, "name": "Spicy Ramen", "emoji": "🍜", "veg": False, "type": "comfort"}, {"id": 17, "name": "Pho", "emoji": "🍲", "veg": False, "type": "comfort"}, {"id": 18, "name": "Pad Thai", "emoji": "🍝", "veg": False, "type": "adventurous"}, {"id": 19, "name": "Korean BBQ", "emoji": "🥩", "veg": False, "type": "adventurous"}, {"id": 20, "name": "Kimchi Fried Rice", "emoji": "🍚", "veg": True, "type": "comfort"}, {"id": 21, "name": "Mapo Tofu", "emoji": "🌶️", "veg": True, "type": "adventurous"}, {"id": 22, "name": "Pork Dumplings", "emoji": "🥟", "veg": False, "type": "comfort"}, {"id": 23, "name": "Miso Mushroom Noodles", "emoji": "🍜", "veg": True, "type": "comfort"}, {"id": 24, "name": "Sichuan Noodles", "emoji": "🍜", "veg": False, "type": "adventurous"}, {"id": 25, "name": "Bibimbap", "emoji": "🥘", "veg": True, "type": "healthy"}, {"id": 26, "name": "Tempura Udon", "emoji": "🍤", "veg": False, "type": "comfort"}, {"id": 27, "name": "Gyoza", "emoji": "🥟", "veg": False, "type": "comfort"}, {"id": 28, "name": "Tom Yum Soup", "emoji": "🍲", "veg": False, "type": "adventurous"}, {"id": 29, "name": "Laksa", "emoji": "🍜", "veg": False, "type": "adventurous"}, {"id": 30, "name": "Nasi Goreng", "emoji": "🍚", "veg": False, "type": "comfort"}, {"id": 31, "name": "Banh Mi", "emoji": "🥖", "veg": False, "type": "adventurous"}, {"id": 32, "name": "Yakitori", "emoji": "🍢", "veg": False, "type": "comfort"}, {"id": 33, "name": "Katsu Curry", "emoji": "🍛", "veg": False, "type": "comfort"}, {"id": 34, "name": "Bulgogi Beef", "emoji": "🥩", "veg": False, "type": "adventurous"}, {"id": 35, "name": "Egg Fried Rice", "emoji": "🍚", "veg": True, "type": "comfort"}, {"id": 36, "name": "Vegetable Spring Rolls", "emoji": "🥢", "veg": True, "type": "adventurous"}, {"id": 37, "name": "Mango Sticky Rice", "emoji": "🥭", "veg": True, "type": "reward"}, {"id": 38, "name": "Margherita Pizza", "emoji": "🍕", "veg": True, "type": "comfort"}, {"id": 39, "name": "Spaghetti Carbonara", "emoji": "🍝", "veg": False, "type": "comfort"}, {"id": 40, "name": "Beef Lasagna", "emoji": "🧀", "veg": False, "type": "comfort"}, {"id": 41, "name": "Mushroom Risotto", "emoji": "🍚", "veg": True, "type": "comfort"}, {"id": 42, "name": "Caesar Salad", "emoji": "🥗", "veg": False, "type": "healthy"}, {"id": 43, "name": "Greek Salad", "emoji": "🥗", "veg": True, "type": "healthy"}, {"id": 44, "name": "Falafel Pita", "emoji": "🥙", "veg": True, "type": "healthy"}, {"id": 45, "name": "Cheeseburger", "emoji": "🍔", "veg": False, "type": "comfort"}, {"id": 46, "name": "Mac & Cheese", "emoji": "🧀", "veg": True, "type": "comfort"}, {"id": 47, "name": "Buffalo Wings", "emoji": "🍗", "veg": False, "type": "comfort"}, {"id": 48, "name": "Hotdog", "emoji": "🌭", "veg": False, "type": "comfort"}, {"id": 49, "name": "Fried Chicken", "emoji": "🍗", "veg": False, "type": "comfort"}, {"id": 50, "name": "Fish and Chips", "emoji": "🐟", "veg": False, "type": "comfort"}, {"id": 51, "name": "Steak", "emoji": "🥩", "veg": False, "type": "reward"}, {"id": 52, "name": "BBQ Ribs", "emoji": "🍖", "veg": False, "type": "adventurous"}, {"id": 53, "name": "Club Sandwich", "emoji": "🥪", "veg": False, "type": "comfort"}, {"id": 54, "name": "Spaghetti & Meatballs", "emoji": "🍝", "veg": False, "type": "comfort"}, {"id": 55, "name": "Beef Tacos", "emoji": "🌮", "veg": False, "type": "adventurous"}, {"id": 56, "name": "Chicken Burrito", "emoji": "🌯", "veg": False, "type": "comfort"}, {"id": 57, "name": "Cheese Quesadilla", "emoji": "🫓", "veg": True, "type": "comfort"}, {"id": 58, "name": "Poutine", "emoji": "🍟", "veg": False, "type": "adventurous"}, {"id": 59, "name": "Chicken Pot Pie", "emoji": "🥧", "veg": False, "type": "comfort"}, {"id": 60, "name": "Shepherd's Pie", "emoji": "🍲", "veg": False, "type": "comfort"}, {"id": 61, "name": "Grilled Cheese", "emoji": "🥪", "veg": True, "type": "comfort"}, {"id": 62, "name": "BLT Sandwich", "emoji": "🥪", "veg": False, "type": "comfort"}, {"id": 63, "name": "Avocado Grain Bowl", "emoji": "🥑", "veg": True, "type": "healthy"}, {"id": 64, "name": "Salmon Rice Bowl", "emoji": "🍣", "veg": False, "type": "healthy"}, {"id": 65, "name": "Burrito Bowl", "emoji": "🌯", "veg": False, "type": "healthy"}, {"id": 66, "name": "Smoothie Bowl", "emoji": "🥤", "veg": True, "type": "healthy"}, {"id": 67, "name": "Greek Yogurt", "emoji": "🥣", "veg": True, "type": "healthy"}, {"id": 68, "name": "Omelette", "emoji": "🍳", "veg": True, "type": "healthy"}, {"id": 69, "name": "Quinoa Salad", "emoji": "🥗", "veg": True, "type": "healthy"}, {"id": 70, "name": "Tuna Poke Bowl", "emoji": "🐟", "veg": False, "type": "healthy"}, {"id": 71, "name": "Grilled Chicken Breast", "emoji": "🍗", "veg": False, "type": "healthy"}, {"id": 72, "name": "Tofu Veggie Bowl", "emoji": "🥘", "veg": True, "type": "healthy"}, {"id": 73, "name": "Veggie Wrap", "emoji": "🌯", "veg": True, "type": "healthy"}, {"id": 74, "name": "Lentil Soup", "emoji": "🍲", "veg": True, "type": "healthy"}, {"id": 75, "name": "Chia Seed Pudding", "emoji": "🥣", "veg": True, "type": "healthy"}, {"id": 76, "name": "Fruit Salad", "emoji": "🍉", "veg": True, "type": "healthy"}, {"id": 77, "name": "Egg White Bites", "emoji": "🥚", "veg": True, "type": "healthy"}, {"id": 78, "name": "Tomato Soup & Toast", "emoji": "🍅", "veg": True, "type": "comfort"}, {"id": 79, "name": "Chicken Congee", "emoji": "🥣", "veg": False, "type": "comfort"}, {"id": 80, "name": "Clam Chowder", "emoji": "🥣", "veg": False, "type": "comfort"}, {"id": 81, "name": "Minestrone Soup", "emoji": "🍲", "veg": True, "type": "comfort"}, {"id": 82, "name": "Chicken Noodle Soup", "emoji": "🍜", "veg": False, "type": "comfort"}, {"id": 83, "name": "Chocolate Cake", "emoji": "🍰", "veg": True, "type": "reward"}, {"id": 84, "name": "Matcha Ice Cream", "emoji": "🍦", "veg": True, "type": "reward"}, {"id": 85, "name": "Bubble Tea", "emoji": "🧋", "veg": True, "type": "reward"}, {"id": 86, "name": "Blueberry Muffin", "emoji": "🧁", "veg": True, "type": "reward"}, {"id": 87, "name": "Pancakes", "emoji": "🥞", "veg": True, "type": "reward"}, {"id": 88, "name": "New York Cheesecake", "emoji": "🍰", "veg": True, "type": "reward"}, {"id": 89, "name": "Fudge Brownie", "emoji": "🍫", "veg": True, "type": "reward"}, {"id": 90, "name": "Apple Pie", "emoji": "🥧", "veg": True, "type": "reward"}, {"id": 91, "name": "Glazed Donut", "emoji": "🍩", "veg": True, "type": "reward"}, {"id": 92, "name": "Tiramisu", "emoji": "🍰", "veg": True, "type": "reward"}, {"id": 93, "name": "Creme Brulee", "emoji": "🍮", "veg": True, "type": "reward"}, {"id": 94, "name": "Belgian Waffle", "emoji": "🧇", "veg": True, "type": "reward"}, {"id": 95, "name": "Butter Croissant", "emoji": "🥐", "veg": True, "type": "reward"}, {"id": 96, "name": "Cinnamon Roll", "emoji": "🥮", "veg": True, "type": "reward"}, {"id": 97, "name": "Gelato", "emoji": "🍨", "veg": True, "type": "reward"}, {"id": 98, "name": "Red Velvet Cake", "emoji": "🧁", "veg": True, "type": "reward"}, {"id": 99, "name": "Macarons", "emoji": "🍪", "veg": True, "type": "reward"}, {"id": 100, "name": "Churros", "emoji": "🍩", "veg": True, "type": "reward"}
]

MOOD_WEIGHTS = {
    "stressed":   {"comfort": 3.0, "reward": 2.5, "healthy": 0.5, "adventurous": 0.5},
    "happy":      {"comfort": 1.5, "reward": 2.0, "healthy": 3.0, "adventurous": 2.0},
    "tired":      {"comfort": 3.0, "reward": 2.0, "healthy": 1.0, "adventurous": 0.0},
    "cozy":       {"comfort": 3.0, "reward": 2.5, "healthy": 1.0, "adventurous": 0.0},
    "adventurous":{"comfort": 0.5, "reward": 1.0, "healthy": 1.5, "adventurous": 3.0},
    "balanced":   {"comfort": 1.5, "reward": 1.5, "healthy": 3.0, "adventurous": 1.5}
}

# Use default weights for all clusters initially
CLUSTER_WEIGHTS = {
    "C1": MOOD_WEIGHTS,
    "C2": MOOD_WEIGHTS,
    "C3": MOOD_WEIGHTS
}

CACHED_DISHES = None
LAST_CACHE_UPDATE = None

# ==========================================
# 3. ML & CLUSTERING FUNCTIONS
# ==========================================
def assign_cluster(raw_features):
    """Calculates nearest centroid using Z-scores"""
    z_features = []
    for i, val in enumerate(raw_features):
        std = FEATURE_STDS[i] if FEATURE_STDS[i] > 0 else 1
        z_val = (val - FEATURE_MEANS[i]) / std
        z_features.append(z_val)
    
    best_cluster = "C1"
    min_dist = float('inf')
    
    for cluster, centroid in CLUSTER_CENTROIDS.items():
        dist = math.sqrt(sum((z_features[i] - centroid[i])**2 for i in range(len(centroid))))
        if dist < min_dist:
            min_dist = dist
            best_cluster = cluster
            
    return best_cluster

def get_user_cluster(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT cluster FROM user_clusters WHERE user_id = %s', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row: return row[0]
    except:
        pass
    return "C1"

def get_dynamic_dishes():
    global CACHED_DISHES, LAST_CACHE_UPDATE
    now = datetime.now()
    if CACHED_DISHES and LAST_CACHE_UPDATE and (now - LAST_CACHE_UPDATE).seconds < 300:
        return CACHED_DISHES
        
    try:
        conn = get_db_connection()
        df_tags = pd.read_sql_query("SELECT dish_name, tag, COUNT(*) as vote_count FROM food_tags GROUP BY dish_name, tag ORDER BY vote_count DESC", conn)
        conn.close()
    except:
        return SEED_DISHES

    dynamic_dishes = {d["name"]: d.copy() for d in SEED_DISHES}
    if not df_tags.empty:
        top_tags = df_tags.drop_duplicates(subset=['dish_name'])
        for _, row in top_tags.iterrows():
            if row['vote_count'] >= 10 and row['dish_name'] in dynamic_dishes:
                dynamic_dishes[row['dish_name']]["type"] = row['tag']
                
    CACHED_DISHES = list(dynamic_dishes.values())
    LAST_CACHE_UPDATE = now
    return CACHED_DISHES

def get_live_ml_recommendations(user_id, mood, diet, time_of_day, weather):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT user_id, dish_name, rating FROM feedback", conn)
        conn.close()
    except Exception as e:
        df = pd.DataFrame()
    
    dynamic_dishes = get_dynamic_dishes()
    unique_users = df['user_id'].unique() if not df.empty else []
    
    # --- COLD START PATH (Cluster + Taste Profile + Context) ---
    if df.empty or user_id not in unique_users or len(unique_users) < 2:
        cluster = get_user_cluster(user_id)
        cluster_weights = CLUSTER_WEIGHTS.get(cluster, MOOD_WEIGHTS)
        mood_weights = cluster_weights.get(mood, cluster_weights["balanced"])
        
        # Taste Profile Boost
        user_ratings = df[df['user_id'] == user_id]
        liked_types = []
        for _, row in user_ratings.iterrows():
            if row['rating'] >= 4:
                info = next((d for d in dynamic_dishes if d["name"] == row['dish_name']), None)
                if info: liked_types.append(info['type'])
        top_liked = Counter(liked_types).most_common(1)[0][0] if liked_types else None
        
        scored_pool = []
        for d in dynamic_dishes:
            if diet == 'vegetarian' and not d["veg"]: continue
            
            base_score = mood_weights.get(d["type"], 0.5)
            
            # Apply Taste Profile boost
            if top_liked and d["type"] == top_liked:
                base_score *= 1.5
                
            # Context multiplier
            if weather == 'rainy' or time_of_day == 'late_night':
                if d["type"] in ['comfort', 'reward']: base_score *= 1.5
            elif weather == 'sunny' or time_of_day == 'morning':
                if d["type"] == 'healthy': base_score *= 1.5
            
            base_score += random.uniform(0, 0.1) 
            scored_pool.append((d, base_score))
            
        scored_pool.sort(key=lambda x: x[1], reverse=True)
        picks = [item[0] for item in scored_pool[:3]]
        
        results = [{"id": d["id"], "name": d["name"], "emoji": d["emoji"], "veg": d["veg"], "desc": ""} for d in picks]
        return results, f"Cold Start: Cluster {cluster} + Taste Profile + Context"

    # --- WARM START PATH (Collaborative Filtering) ---
    matrix = df.pivot_table(index='user_id', columns='dish_name', values='rating')
    matrix_filled = matrix.apply(lambda row: row.fillna(row.mean()), axis=1).fillna(0)
    
    user_similarity = cosine_similarity(matrix_filled)
    sim_df = pd.DataFrame(user_similarity, index=matrix.index, columns=matrix.index)
    
    user_sims = sim_df[user_id].drop(user_id)
    user_rated_items = matrix.loc[user_id].dropna().index
    all_items = [d["name"] for d in dynamic_dishes]
    unrated_items = [item for item in all_items if item not in user_rated_items]
    
    predictions = {}
    for item in unrated_items:
        if item not in matrix.columns: continue
        item_ratings = matrix[item].dropna()
        valid_users = [u for u in item_ratings.index if u in user_sims.index]
        if not valid_users: continue
        sim_scores = user_sims[valid_users]
        ratings = item_ratings[valid_users]
        if sim_scores.sum() > 0:
            predictions[item] = (sim_scores * ratings).sum() / sim_scores.sum()
            
    top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    results_list = []
    for item, score in top_items:
        info = next((d for d in dynamic_dishes if d["name"] == item), None)
        if not info: continue
        if diet == 'vegetarian' and not info['veg']: continue
        results_list.append({"id": info["id"], "name": item, "emoji": info["emoji"], "veg": info["veg"], "desc": ""})
        if len(results_list) == 3: break
        
    # Smart padding if CF wasn't enough
    if len(results_list) < 3:
        user_history = df[df['user_id'] == user_id]
        liked_types = []
        for _, row in user_history.iterrows():
            if row['rating'] >= 4:
                dish_info = next((d for d in dynamic_dishes if d["name"] == row['dish_name']), None)
                if dish_info: liked_types.append(dish_info['type'])
        
        if liked_types:
            top_preferred_type = Counter(liked_types).most_common(1)[0][0]
            smart_pool = [d for d in dynamic_dishes if d["type"] == top_preferred_type and d["name"] not in user_rated_items and (diet == 'all' or d["veg"])]
            random.shuffle(smart_pool)
            for d in smart_pool:
                if d["name"] not in [r["name"] for r in results_list]:
                    results_list.append({"id": d["id"], "name": d["name"], "emoji": d["emoji"], "veg": d["veg"], "desc": ""})
                if len(results_list) == 3: break
                
        if len(results_list) < 3:
            mood_weights = MOOD_WEIGHTS.get(mood, MOOD_WEIGHTS["balanced"])
            remaining_pool = [d for d in dynamic_dishes if d["name"] not in user_rated_items and d["name"] not in [r["name"] for r in results_list] and (diet == 'all' or d["veg"])]
            
            scored_remaining = []
            for d in remaining_pool:
                score = mood_weights.get(d["type"], 0.5) + random.uniform(0, 0.1)
                scored_remaining.append((d, score))
            
            scored_remaining.sort(key=lambda x: x[1], reverse=True)
            for d, _ in scored_remaining:
                results_list.append({"id": d["id"], "name": d["name"], "emoji": d["emoji"], "veg": d["veg"], "desc": ""})
                if len(results_list) == 3: break

    return results_list, "Collaborative Filtering (Live Cosine Similarity)"

# ==========================================
# 4. FLASK API ROUTES
# ==========================================
@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    data = request.json
    user_id = data.get('user_id')
    ratings = data.get('ratings', [])
    
    # 1. Build the 8-feature vector for K-Means clustering
    raw_features = [
        float(data.get('q1', 3.0)), 
        float(data.get('q2', 3.0)), 
        float(data.get('q3', 3.0)), 
        float(data.get('q4', 0.5)), 
        float(data.get('q5', 0.5)), 
        float(data.get('q6', 0.5)), 
        float(data.get('q7', 0.5)), 
        float(data.get('q8', 3.0))
    ]
    
    # 2. Calculate Z-scores and assign cluster
    assigned_cluster = assign_cluster(raw_features)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 3. Save the cluster to DB
        cursor.execute('''
            INSERT INTO user_clusters (user_id, cluster) 
            VALUES (%s, %s) 
            ON CONFLICT (user_id) DO UPDATE SET cluster = EXCLUDED.cluster
        ''', (user_id, assigned_cluster))
        
        # 4. Save the food ratings (Taste Profile) to DB
        for item in ratings:
            cursor.execute('''
                INSERT INTO feedback (user_id, dish_name, rating, mood) 
                VALUES (%s, %s, %s, 'onboarding')
            ''', (user_id, item['dish'], item['rating']))
            
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"status": "success", "message": f"Profile built! Assigned to {assigned_cluster}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def recommend():
    data = request.json
    recs, logic_used = get_live_ml_recommendations(
        data.get('user_id', 'Anonymous'), 
        data.get('mood', 'stressed'), 
        data.get('diet', 'all'),
        data.get('time', 'afternoon'), 
        data.get('weather', 'sunny')
    )
    return jsonify({"recommendation_id": f"rec_{datetime.now().timestamp()}", "cluster_logic": logic_used, "dishes": recs})

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    rating_map = {"yes": 5, "okay": 3, "no": 1}
    numeric_rating = rating_map.get(data.get('rating'), 3)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedback (user_id, dish_name, rating, mood) VALUES (%s, %s, %s, %s)', 
                       (data.get('user_id', 'Anonymous'), data.get('dish_name'), numeric_rating, data.get('mood')))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "success", "message": "Live ML model updated!"})

@app.route('/api/tag_food', methods=['POST'])
def tag_food():
    data = request.json
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO food_tags (dish_name, tag, user_id) VALUES (%s, %s, %s)', 
                       (data.get('dish_name'), data.get('tag'), data.get('user_id', 'Anonymous')))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"status": "success", "message": "Tag saved! The AI will dynamically update."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
