from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import io
import zipfile
import random
import sqlite3
import base64
from datetime import datetime
import json
import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- ROBUST PATH SETUP ---
# Explicitly tell Flask where to find folders to prevent TemplateNotFound on Render
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
load_dotenv() 

# --- AI SETUP ---
try:
    # Use the GEMINI_API_KEY from your Render Environment Variables
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"AI Client Setup Error: {e}")
    client = None

# --- DATABASE SETUP ---
def init_db():
    db_path = os.path.join(base_dir, 'design_logs.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            category TEXT,
            vibe TEXT,
            layout TEXT,
            font TEXT,
            primary_color TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- LOAD ML MODEL ---
try:
    model_path = os.path.join(base_dir, 'model_brain.pkl')
    with open(model_path, 'rb') as f:
        brain = pickle.load(f)
except FileNotFoundError:
    print("Error: model_brain.pkl not found in root directory!")
    exit()

CATEGORIES = [
    'Automotive', 'Beauty', 'Construction', 'Cybersecurity', 'Education', 
    'Entertainment', 'Fashion', 'Finance', 'Food', 'Gaming', 
    'Healthcare', 'Interior', 'Legal', 'Marketing', 'Music', 
    'Non-Profit', 'Photography', 'Real Estate', 'Sports', 'Technology', 'Travel'
]
VIBES = ['Commercial', 'Promotional', 'Balanced', 'Corporate', 'Professional']
cat_map = {k: v for v, k in enumerate(CATEGORIES)}
vibe_map = {k: v for v, k in enumerate(VIBES)}

# --- UNSPLASH INTEGRATION ---
unsplash_cache = {} 

def get_dynamic_unsplash_image(category):
    fallback_image = 'https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=1600&q=80'
    
    if category in unsplash_cache and len(unsplash_cache[category]) > 0:
        return random.choice(unsplash_cache[category])

    api_key = os.getenv("UNSPLASH_API_KEY")
    if not api_key: return fallback_image
    
    url = f"https://api.unsplash.com/search/photos?query={category} industry&client_id={api_key}&orientation=landscape&per_page=15"
    try:
        response = requests.get(url, timeout=2.0) 
        data = response.json()
        
        if data.get('results'):
            urls = [res['urls']['regular'] for res in data['results']]
            unsplash_cache[category] = urls
            return random.choice(urls)
            
        return fallback_image
    except Exception as e:
        print("Unsplash Error:", e)
        return fallback_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        cat = data.get('category')
        vibe_index = int(data.get('vibe_value', 2))
        vibe = VIBES[max(0, min(vibe_index, 4))]
        variation = int(data.get('variation', 0)) % 5 

        input_df = pd.DataFrame([[cat_map.get(cat, 0), vibe_map.get(vibe, 0), variation]], 
                                columns=['Cat_Code', 'Vibe_Code', 'Variation'])
        pred = brain.predict(input_df)[0]
        
        img_url = get_dynamic_unsplash_image(cat)

        return jsonify({
            'category': cat,
            'layout': pred[0], 
            'font': pred[1],
            'primary': pred[2], 
            'secondary': pred[3], 
            'tertiary': pred[4],
            'headline': f"Redefining {cat}",
            'sub': f"Experience the future of the {cat} industry with our AI-driven solutions.",
            'vibe': vibe,
            'img_url': img_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/suggest', methods=['POST'])
def suggest():
    if not client: return jsonify({"error": "AI Client not initialized"}), 500
    data = request.json
    prompt = f"Expert UX advice for {data.get('category')} kit. Vibe: {data.get('vibe')}. Layout: {data.get('layout')}. Colors: {data.get('primary')}, {data.get('secondary')}. Provide 3 actionable points in JSON format like this: {{\"suggestions\": [\"point 1\", \"point 2\", \"point 3\"]}}"
    try:
        # Fixed model name to a valid production version
        response = client.models.generate_content(
            model='gemini-1.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return jsonify(json.loads(response.text))
    except Exception as e:
        print(f"Gemini AI Error: {e}")
        return jsonify({"error": "Failed to generate suggestions"}), 500

# --- EXPORT HELPERS ---
def generate_html(data):
    raw_font = data.get('font', 'Inter')
    clean_font = raw_font.split('(')[0].strip()
    font_url = clean_font.replace(' ', '+')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.get('category')} AI Design</title>
    <link href="https://fonts.googleapis.com/css2?family={font_url}:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        primary: '{data.get('primary')}',
                        secondary: '{data.get('secondary')}',
                        tertiary: '{data.get('tertiary')}',
                    }},
                    fontFamily: {{
                        sans: ['"{clean_font}"', 'sans-serif'],
                    }}
                }}
            }}
        }}
    </script>
</head>
<body class="bg-secondary text-primary font-sans antialiased">
    <nav class="bg-tertiary text-white shadow-md relative z-50">
        <div class="container mx-auto px-6 py-4 flex justify-between items-center">
            <h1 class="text-2xl font-bold tracking-tight">{data.get('category')}<span class="opacity-70">.</span></h1>
            <button id="mobile-menu-btn" class="md:hidden">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
            </button>
            <div class="hidden md:flex space-x-8 items-center">
                <a href="#" class="hover:opacity-80">Home</a>
                <a href="#" class="hover:opacity-80">About</a>
                <a href="#" class="bg-white text-tertiary px-5 py-2 rounded-full font-semibold">Contact Us</a>
            </div>
        </div>
    </nav>
    <header class="container mx-auto px-6 py-16 text-center">
        <h2 class="text-4xl md:text-6xl font-extrabold mb-6">{data.get('headline')}</h2>
        <p class="text-lg opacity-80 max-w-2xl mx-auto mb-10">{data.get('sub')}</p>
        <img src="{data.get('img_url')}" class="rounded-2xl shadow-2xl mx-auto max-h-[500px]">
    </header>
    <script>
        document.getElementById('mobile-menu-btn').addEventListener('click', () => alert('Menu toggled!'));
    </script>
</body>
</html>"""

def generate_css(data):
    return f"/* Custom CSS */\n::-webkit-scrollbar {{ width: 10px; }}\n::-webkit-scrollbar-thumb {{ background: {data.get('tertiary')}; }}"

def generate_js(data):
    return f"console.log('{data.get('category')} design kit loaded.');"

@app.route('/view-code', methods=['POST'])
def view_code():
    data = request.json
    return jsonify({
        "html": generate_html(data),
        "css": generate_css(data),
        "js": generate_js(data)
    })

@app.route('/download', methods=['POST'])
def download():
    data = request.json
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('index.html', generate_html(data))
        zf.writestr('style.css', generate_css(data))
        zf.writestr('script.js', generate_js(data))
    memory_file.seek(0)
    return send_file(memory_file, download_name='ai_design_kit.zip', as_attachment=True)

if __name__ == '__main__':
    # Use environment port for local testing, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
