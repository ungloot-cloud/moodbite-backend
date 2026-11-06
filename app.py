from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import psycopg2
import random

app = Flask(__name__)
CORS(app) 

# 你的 Neon 数据库连接信息（已直接填入）
DEFAULT_DB_URL = "postgresql://neondb_owner:npg_aMHZtF4wo2nK@ep-tiny-thunder-aos3eit7.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

# --- 1. 初始化 PostgreSQL 数据库表 ---
def init_db():
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
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ 成功连接到 Neon PostgreSQL 数据库")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")

init_db()

# --- 2. 食物数据库 ---
DISHES = [
    {"name": "Miso mushroom noodles", "emoji": "🍜", "veg": True, "type": "comfort"},
    {"name": "Tomato soup & toast", "emoji": "🍅", "veg": True, "type": "comfort"},
    {"name": "Spicy Ramen", "emoji": "🍜", "veg": False, "type": "comfort"},
    {"name": "Pork Dumplings", "emoji": "🥟", "veg": False, "type": "comfort"},
    {"name": "Mac & Cheese", "emoji": "🧀", "veg": True, "type": "comfort"},
    {"name": "Margherita Pizza", "emoji": "🍕", "veg": True, "type": "comfort"},
    {"name": "Beef Burger", "emoji": "🍔", "veg": False, "type": "comfort"},
    {"name": "Pasta Carbonara", "emoji": "🍝", "veg": False, "type": "comfort"},
    {"name": "Chicken Congee", "emoji": "🥣", "veg": False, "type": "comfort"},
    {"name": "Fish and Chips", "emoji": "🐟", "veg": False, "type": "comfort"},
    {"name": "Pho", "emoji": "🍲", "veg": False, "type": "comfort"},
    {"name": "Naan and Curry", "emoji": "🍛", "veg": True, "type": "comfort"},
    {"name": "Avocado grain bowl", "emoji": "🥑", "veg": True, "type": "healthy"},
    {"name": "Salmon rice bowl", "emoji": "🍣", "veg": False, "type": "healthy"},
    {"name": "Falafel pita", "emoji": "🥙", "veg": True, "type": "healthy"},
    {"name": "Caesar salad", "emoji": "🥗", "veg": False, "type": "healthy"},
    {"name": "Sushi platter", "emoji": "🍱", "veg": False, "type": "healthy"},
    {"name": "Greek yogurt", "emoji": "🥣", "veg": True, "type": "healthy"},
    {"name": "Omelette", "emoji": "🍳", "veg": True, "type": "healthy"},
    {"name": "Smoothie bowl", "emoji": "🥤", "veg": True, "type": "healthy"},
    {"name": "Burrito bowl", "emoji": "🌯", "veg": False, "type": "healthy"},
    {"name": "Spicy chicken tacos", "emoji": "🌮", "veg": False, "type": "adventurous"},
    {"name": "Kimchi tofu bibimbap", "emoji": "🥘", "veg": True, "type": "adventurous"},
    {"name": "Pad Thai", "emoji": "🍝", "veg": False, "type": "adventurous"},
    {"name": "Mapo Tofu", "emoji": "🌶️", "veg": True, "type": "adventurous"},
    {"name": "Sichuan Noodles", "emoji": "🍜", "veg": False, "type": "adventurous"},
    {"name": "Chocolate cake", "emoji": "🍰", "veg": True, "type": "reward"},
    {"name": "Matcha Ice cream", "emoji": "🍦", "veg": True, "type": "reward"},
    {"name": "Bubble tea", "emoji": "🧋", "veg": True, "type": "reward"},
    {"name": "Blueberry muffin", "emoji": "🧁", "veg": True, "type": "reward"},
    {"name": "Pancakes", "emoji": "🥞", "veg": True, "type": "reward"},
    {"name": "Steak", "emoji": "🥩", "veg": False, "type": "reward"}
]

dish_lookup = {d["name"]: {"id": i+1, "emoji": d["emoji"], "veg": d["veg"]} for i, d in enumerate(DISHES)}

def get_live_ml_recommendations(user_id, mood, diet):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT user_id, dish_name, rating FROM feedback", conn)
        conn.close()
    except Exception as e:
        print(f"数据库读取失败: {e}")
        df = pd.DataFrame()
    
    # --- COLD START FALLBACK ---
    if df.empty or user_id not in df['user_id'].values:
        comfort_moods = ['stressed', 'tired', 'cozy']
        if mood in comfort_moods: valid_types = ['comfort', 'reward']
        elif mood in ['happy', 'balanced']: valid_types = ['healthy', 'comfort']
        else: valid_types = ['adventurous', 'healthy']
            
        pool = [d for d in DISHES if d["type"] in valid_types]
        if diet == 'vegetarian': pool = [d for d in pool if d["veg"]]
            
        picks = random.sample(pool, min(3, len(pool)))
        results = []
        for d in picks:
            info = dish_lookup[d["name"]]
            results.append({
                "id": info["id"], "name": d["name"], "emoji": info["emoji"], "veg": info["veg"],
                "desc": f"A great match for your {mood} mood."
            })
        return results, "Cold Start: Mood-Based Clustering Applied"

    # --- WARM START: LIVE COLLABORATIVE FILTERING ---
    matrix = df.pivot_table(index='user_id', columns='dish_name', values='rating')
    matrix_filled = matrix.fillna(0)
    user_similarity = cosine_similarity(matrix_filled)
    sim_df = pd.DataFrame(user_similarity, index=matrix.index, columns=matrix.index)
    
    user_sims = sim_df[user_id].drop(user_id)
    user_rated_items = matrix.loc[user_id].dropna().index
    all_items = list(dish_lookup.keys())
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
            predicted_rating = (sim_scores * ratings).sum() / sim_scores.sum()
            predictions[item] = predicted_rating
            
    if len(predictions) < 3:
        return get_live_ml_recommendations("NEW_USER_" + user_id, mood, diet)
        
    top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    results = []
    for item, score in top_items:
        info = dish_lookup.get(item)
        if not info: continue
        if diet == 'vegetarian' and not info['veg']: continue
        results.append({
            "id": info["id"], "name": item, "emoji": info["emoji"], "veg": info["veg"],
            "desc": f"AI Prediction: {score:.1f}/5 rating based on your taste profile."
        })
        if len(results) == 3: break
    return results, "Collaborative Filtering (Live Cosine Similarity)"

@app.route('/api/recommendations', methods=['POST'])
def recommend():
    data = request.json
    dishes, logic_used = get_live_ml_recommendations(
        data.get('user_id', 'Anonymous'), 
        data.get('mood', 'stressed'), 
        data.get('diet', 'all')
    )
    return jsonify({
        "recommendation_id": f"rec_{datetime.now().timestamp()}",
        "cluster_logic": logic_used,
        "dishes": dishes
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    rating_map = {"yes": 5, "okay": 3, "no": 1}
    numeric_rating = rating_map.get(data.get('rating'), 3)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (user_id, dish_name, rating, mood)
            VALUES (%s, %s, %s, %s)
        ''', (data.get('user_id', 'Anonymous'), data.get('dish_name'), numeric_rating, data.get('mood')))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"写入数据库失败: {e}")
        return jsonify({"status": "error", "message": "Database write failed"}), 500
        
    return jsonify({"status": "success", "message": "Live ML model updated!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
