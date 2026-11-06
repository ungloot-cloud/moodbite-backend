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

# Neon 数据库连接信息
DEFAULT_DB_URL = "postgresql://neondb_owner:npg_aMHZtF4wo2nK@ep-tiny-thunder-aos3eit7.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

# 1. 初始化 PostgreSQL 数据库表
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
        print("✅ 成功连接并初始化 Neon PostgreSQL 数据库")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")

init_db()

# 2. 食物数据库 (100 Items - 与前端完全匹配)
DISHES = [
    # --- Hong Kong / Cantonese ---
    {"name": "Dim Sum Platter", "emoji": "🥟", "veg": False, "type": "comfort"},
    {"name": "Wonton Noodles", "emoji": "🍜", "veg": False, "type": "comfort"},
    {"name": "Roast Goose Rice", "emoji": "🦆", "veg": False, "type": "comfort"},
    {"name": "Pineapple Bun", "emoji": "🍞", "veg": True, "type": "reward"},
    {"name": "Egg Tart", "emoji": "🥧", "veg": True, "type": "reward"},
    {"name": "HK Milk Tea", "emoji": "🧋", "veg": True, "type": "reward"},
    {"name": "Beef Brisket Noodles", "emoji": "🍜", "veg": False, "type": "comfort"},
    {"name": "Claypot Rice", "emoji": "🍚", "veg": False, "type": "comfort"},
    {"name": "Cheung Fun", "emoji": "🥢", "veg": True, "type": "comfort"},
    {"name": "Curry Fish Balls", "emoji": "🍢", "veg": False, "type": "adventurous"},
    {"name": "Siu Mei Platter", "emoji": "🍖", "veg": False, "type": "comfort"},
    {"name": "HK French Toast", "emoji": "🍞", "veg": True, "type": "reward"},
    {"name": "Red Bean Soup", "emoji": "🍲", "veg": True, "type": "reward"},
    {"name": "Mango Pomelo Sago", "emoji": "🥭", "veg": True, "type": "reward"},
    
    # --- Asian ---
    {"name": "Sushi Platter", "emoji": "🍣", "veg": False, "type": "healthy"},
    {"name": "Spicy Ramen", "emoji": "🍜", "veg": False, "type": "comfort"},
    {"name": "Pho", "emoji": "🍲", "veg": False, "type": "comfort"},
    {"name": "Pad Thai", "emoji": "🍝", "veg": False, "type": "adventurous"},
    {"name": "Korean BBQ", "emoji": "🥩", "veg": False, "type": "adventurous"},
    {"name": "Kimchi Fried Rice", "emoji": "🍚", "veg": True, "type": "comfort"},
    {"name": "Mapo Tofu", "emoji": "🌶️", "veg": True, "type": "adventurous"},
    {"name": "Pork Dumplings", "emoji": "🥟", "veg": False, "type": "comfort"},
    {"name": "Miso Mushroom Noodles", "emoji": "🍜", "veg": True, "type": "comfort"},
    {"name": "Sichuan Noodles", "emoji": "🍜", "veg": False, "type": "adventurous"},
    {"name": "Bibimbap", "emoji": "🥘", "veg": True, "type": "healthy"},
    {"name": "Tempura Udon", "emoji": "🍤", "veg": False, "type": "comfort"},
    {"name": "Gyoza", "emoji": "🥟", "veg": False, "type": "comfort"},
    {"name": "Tom Yum Soup", "emoji": "🍲", "veg": False, "type": "adventurous"},
    {"name": "Laksa", "emoji": "🍜", "veg": False, "type": "adventurous"},
    {"name": "Nasi Goreng", "emoji": "🍚", "veg": False, "type": "comfort"},
    {"name": "Banh Mi", "emoji": "🥖", "veg": False, "type": "adventurous"},
    {"name": "Yakitori", "emoji": "🍢", "veg": False, "type": "comfort"},
    {"name": "Katsu Curry", "emoji": "🍛", "veg": False, "type": "comfort"},
    {"name": "Bulgogi Beef", "emoji": "🥩", "veg": False, "type": "adventurous"},
    {"name": "Egg Fried Rice", "emoji": "🍚", "veg": True, "type": "comfort"},
    {"name": "Vegetable Spring Rolls", "emoji": "🥢", "veg": True, "type": "adventurous"},
    {"name": "Mango Sticky Rice", "emoji": "🥭", "veg": True, "type": "reward"},
    
    # --- Western / American ---
    {"name": "Margherita Pizza", "emoji": "🍕", "veg": True, "type": "comfort"},
    {"name": "Spaghetti Carbonara", "emoji": "🍝", "veg": False, "type": "comfort"},
    {"name": "Beef Lasagna", "emoji": "🧀", "veg": False, "type": "comfort"},
    {"name": "Mushroom Risotto", "emoji": "🍚", "veg": True, "type": "comfort"},
    {"name": "Caesar Salad", "emoji": "🥗", "veg": False, "type": "healthy"},
    {"name": "Greek Salad", "emoji": "🥗", "veg": True, "type": "healthy"},
    {"name": "Falafel Pita", "emoji": "🥙", "veg": True, "type": "healthy"},
    {"name": "Cheeseburger", "emoji": "🍔", "veg": False, "type": "comfort"},
    {"name": "Mac & Cheese", "emoji": "🧀", "veg": True, "type": "comfort"},
    {"name": "Buffalo Wings", "emoji": "🍗", "veg": False, "type": "comfort"},
    {"name": "Hotdog", "emoji": "🌭", "veg": False, "type": "comfort"},
    {"name": "Fried Chicken", "emoji": "🍗", "veg": False, "type": "comfort"},
    {"name": "Fish and Chips", "emoji": "🐟", "veg": False, "type": "comfort"},
    {"name": "Steak", "emoji": "🥩", "veg": False, "type": "reward"},
    {"name": "BBQ Ribs", "emoji": "🍖", "veg": False, "type": "adventurous"},
    {"name": "Club Sandwich", "emoji": "🥪", "veg": False, "type": "comfort"},
    {"name": "Spaghetti & Meatballs", "emoji": "🍝", "veg": False, "type": "comfort"},
    {"name": "Beef Tacos", "emoji": "🌮", "veg": False, "type": "adventurous"},
    {"name": "Chicken Burrito", "emoji": "🌯", "veg": False, "type": "comfort"},
    {"name": "Cheese Quesadilla", "emoji": "🫓", "veg": True, "type": "comfort"},
    {"name": "Poutine", "emoji": "🍟", "veg": False, "type": "adventurous"},
    {"name": "Chicken Pot Pie", "emoji": "🥧", "veg": False, "type": "comfort"},
    {"name": "Shepherd's Pie", "emoji": "🍲", "veg": False, "type": "comfort"},
    {"name": "Grilled Cheese", "emoji": "🥪", "veg": True, "type": "comfort"},
    {"name": "BLT Sandwich", "emoji": "🥪", "veg": False, "type": "comfort"},
    
    # --- Healthy / Bowls ---
    {"name": "Avocado Grain Bowl", "emoji": "🥑", "veg": True, "type": "healthy"},
    {"name": "Salmon Rice Bowl", "emoji": "🍣", "veg": False, "type": "healthy"},
    {"name": "Burrito Bowl", "emoji": "🌯", "veg": False, "type": "healthy"},
    {"name": "Smoothie Bowl", "emoji": "🥤", "veg": True, "type": "healthy"},
    {"name": "Greek Yogurt", "emoji": "🥣", "veg": True, "type": "healthy"},
    {"name": "Omelette", "emoji": "🍳", "veg": True, "type": "healthy"},
    {"name": "Quinoa Salad", "emoji": "🥗", "veg": True, "type": "healthy"},
    {"name": "Tuna Poke Bowl", "emoji": "🐟", "veg": False, "type": "healthy"},
    {"name": "Grilled Chicken Breast", "emoji": "🍗", "veg": False, "type": "healthy"},
    {"name": "Tofu Veggie Bowl", "emoji": "🥘", "veg": True, "type": "healthy"},
    {"name": "Veggie Wrap", "emoji": "🌯", "veg": True, "type": "healthy"},
    {"name": "Lentil Soup", "emoji": "🍲", "veg": True, "type": "healthy"},
    {"name": "Chia Seed Pudding", "emoji": "🥣", "veg": True, "type": "healthy"},
    {"name": "Fruit Salad", "emoji": "🍉", "veg": True, "type": "healthy"},
    {"name": "Egg White Bites", "emoji": "🥚", "veg": True, "type": "healthy"},
    
    # --- Comfort / Soups ---
    {"name": "Tomato Soup & Toast", "emoji": "🍅", "veg": True, "type": "comfort"},
    {"name": "Chicken Congee", "emoji": "🥣", "veg": False, "type": "comfort"},
    {"name": "Clam Chowder", "emoji": "🥣", "veg": False, "type": "comfort"},
    {"name": "Minestrone Soup", "emoji": "🍲", "veg": True, "type": "comfort"},
    {"name": "Chicken Noodle Soup", "emoji": "🍜", "veg": False, "type": "comfort"},
    
    # --- Reward / Desserts / Snacks ---
    {"name": "Chocolate Cake", "emoji": "🍰", "veg": True, "type": "reward"},
    {"name": "Matcha Ice Cream", "emoji": "🍦", "veg": True, "type": "reward"},
    {"name": "Bubble Tea", "emoji": "🧋", "veg": True, "type": "reward"},
    {"name": "Blueberry Muffin", "emoji": "🧁", "veg": True, "type": "reward"},
    {"name": "Pancakes", "emoji": "🥞", "veg": True, "type": "reward"},
    {"name": "New York Cheesecake", "emoji": "🍰", "veg": True, "type": "reward"},
    {"name": "Fudge Brownie", "emoji": "🍫", "veg": True, "type": "reward"},
    {"name": "Apple Pie", "emoji": "🥧", "veg": True, "type": "reward"},
    {"name": "Glazed Donut", "emoji": "🍩", "veg": True, "type": "reward"},
    {"name": "Tiramisu", "emoji": "🍰", "veg": True, "type": "reward"},
    {"name": "Creme Brulee", "emoji": "🍮", "veg": True, "type": "reward"},
    {"name": "Belgian Waffle", "emoji": "🧇", "veg": True, "type": "reward"},
    {"name": "Butter Croissant", "emoji": "🥐", "veg": True, "type": "reward"},
    {"name": "Cinnamon Roll", "emoji": "🥮", "veg": True, "type": "reward"},
    {"name": "Gelato", "emoji": "🍨", "veg": True, "type": "reward"},
    {"name": "Red Velvet Cake", "emoji": "🧁", "veg": True, "type": "reward"},
    {"name": "Macarons", "emoji": "🍪", "veg": True, "type": "reward"},
    {"name": "Churros", "emoji": "🍩", "veg": True, "type": "reward"},
    {"name": "Caramel Pudding", "emoji": "🍮", "veg": True, "type": "reward"},
    {"name": "Chocolate Eclair", "emoji": "🥐", "veg": True, "type": "reward"},
    {"name": "Spicy Chicken Tacos", "emoji": "🌮", "veg": False, "type": "adventurous"},
    {"name": "Naan and Curry", "emoji": "🍛", "veg": True, "type": "comfort"},
    {"name": "Samosa", "emoji": "🥟", "veg": True, "type": "reward"}
]

# 生成 ID 映射 (1 到 100)
dish_lookup = {d["name"]: {"id": i+1, "emoji": d["emoji"], "veg": d["veg"]} for i, d in enumerate(DISHES)}

# 3. 核心算法逻辑
def get_live_ml_recommendations(user_id, mood, diet):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT user_id, dish_name, rating FROM feedback", conn)
        conn.close()
    except Exception as e:
        print(f"DB Read Error: {e}")
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
            
    # 如果预测数量少于3个，退回 Cold Start
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
        
    # 如果过滤后依然不足3个（例如素食者选项太少），退回 Cold Start
    if len(results) < 3:
        return get_live_ml_recommendations("NEW_USER_" + user_id, mood, diet)
        
    return results, "Collaborative Filtering (Live Cosine Similarity)"

# 4. API Routes
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
        print(f"DB Write Error: {e}")
        return jsonify({"status": "error", "message": "Database write failed"}), 500
        
    return jsonify({"status": "success", "message": "Live ML model updated!"})

# 开发者测试接口 (保留在后台，不影响前端)
@app.route('/api/debug_database/<user_id>', methods=['GET'])
def debug_database(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, dish_name, rating, mood, timestamp FROM feedback WHERE user_id = %s ORDER BY timestamp DESC LIMIT 10", (user_id,))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "user_id": row[0],
                "dish_name": row[1],
                "rating": row[2],
                "mood": row[3],
                "timestamp": str(row[4])
            })
            
        cursor.close()
        conn.close()
        return jsonify({"status": "success", "database": "Neon PostgreSQL", "rows": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
