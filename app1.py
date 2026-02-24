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

app = Flask(__name__)
load_dotenv() 

# --- AI SETUP ---
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except:
    client = None

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('design_logs.db')
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
    with open('model_brain.pkl', 'rb') as f:
        brain = pickle.load(f)
except FileNotFoundError:
    print("Error: Run Train_model.py first!")
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

# --- UNSPLASH INTEGRATION (WITH CACHING) ---
unsplash_cache = {} 

def get_dynamic_unsplash_image(category):
    fallback_image = 'https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=1600&q=80'
    
    # 1. Check if we already have images for this category in the cache
    if category in unsplash_cache and len(unsplash_cache[category]) > 0:
        return random.choice(unsplash_cache[category])

    # 2. If not, fetch them from Unsplash
    api_key = os.getenv("UNSPLASH_API_KEY")
    if not api_key: return fallback_image
    
    url = f"https://api.unsplash.com/search/photos?query={category} industry&client_id={api_key}&orientation=landscape&per_page=15"
    try:
        response = requests.get(url, timeout=1.5) 
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
    if not client: return jsonify({"error": "API key missing"}), 500
    data = request.json
    prompt = f"Expert UX advice for {data.get('category')} kit. Vibe: {data.get('vibe')}. Layout: {data.get('layout')}. Colors: {data.get('primary')}, {data.get('secondary')}. Provide 3 actionable points in JSON format like this: {{\"suggestions\": [\"point 1\", \"point 2\", \"point 3\"]}}"
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return jsonify(json.loads(response.text))
    except Exception as e:
        return jsonify({"error": "Failed to generate suggestions"}), 500

# Helper to generate HTML code string (Tailwind Version)
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
    
    <link rel="stylesheet" href="style.css">
</head>
<body class="bg-secondary text-primary font-sans antialiased">
    
    <nav class="bg-tertiary text-white shadow-md relative z-50">
        <div class="container mx-auto px-6 py-4 flex justify-between items-center">
            <h1 class="text-2xl font-bold tracking-tight">{data.get('category')}<span class="opacity-70">.</span></h1>
            
            <button id="mobile-menu-btn" class="md:hidden focus:outline-none">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
                </svg>
            </button>

            <div class="hidden md:flex space-x-8 items-center">
                <a href="#" class="hover:opacity-80 transition">Home</a>
                <a href="#" class="hover:opacity-80 transition">About</a>
                <a href="#" class="hover:opacity-80 transition">Services</a>
                <a href="#" class="bg-white text-tertiary px-5 py-2 rounded-full font-semibold hover:bg-opacity-90 transition">Contact Us</a>
            </div>
        </div>
        
        <div id="mobile-menu" class="hidden md:hidden absolute w-full bg-tertiary border-t border-white/20">
            <a href="#" class="block px-6 py-3 hover:bg-white/10">Home</a>
            <a href="#" class="block px-6 py-3 hover:bg-white/10">About</a>
            <a href="#" class="block px-6 py-3 hover:bg-white/10">Services</a>
            <a href="#" class="block px-6 py-3 hover:bg-white/10 font-bold">Contact Us</a>
        </div>
    </nav>

    <header class="container mx-auto px-6 py-16 md:py-24 text-center">
        <span class="text-tertiary font-semibold tracking-wider uppercase text-sm mb-4 block">Welcome to the future</span>
        <h2 class="text-4xl md:text-6xl font-extrabold mb-6 leading-tight">{data.get('headline')}</h2>
        <p class="text-lg md:text-xl opacity-80 max-w-2xl mx-auto mb-10">{data.get('sub')}</p>
        
        <div class="rounded-2xl overflow-hidden shadow-2xl max-w-5xl mx-auto group">
            <img src="{data.get('img_url')}" alt="Hero" class="w-full h-auto object-cover max-h-[500px] transform group-hover:scale-105 transition duration-700 ease-in-out">
        </div>
        
        <button id="cta-btn" class="mt-12 bg-tertiary text-white px-10 py-4 rounded-full text-lg font-semibold hover:shadow-lg hover:-translate-y-1 transition transform duration-300">
            Explore {data.get('category')} Solutions
        </button>
    </header>

    <script src="script.js"></script>
</body>
</html>"""

# Helper to generate CSS code string
def generate_css(data):
    return f"""/* Custom CSS beyond Tailwind utilities */
html {{
    scroll-behavior: smooth;
}}

/* Add custom scrollbar styling */
::-webkit-scrollbar {{
    width: 10px;
}}
::-webkit-scrollbar-track {{
    background: {data.get('secondary')};
}}
::-webkit-scrollbar-thumb {{
    background: {data.get('tertiary')};
    border-radius: 5px;
}}"""

# Helper to generate JS code string
def generate_js(data):
    return f"""// Interaction Logic for the AI Generated UI Kit
document.addEventListener('DOMContentLoaded', () => {{
    
    // 1. Mobile Menu Toggle
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (mobileBtn && mobileMenu) {{
        mobileBtn.addEventListener('click', () => {{
            mobileMenu.classList.toggle('hidden');
        }});
    }}

    // 2. Call to Action Button Alert
    const ctaBtn = document.getElementById('cta-btn');
    if (ctaBtn) {{
        ctaBtn.addEventListener('click', () => {{
            alert('Welcome to your new {data.get('category')} platform! This button was powered by your generated script.js file.');
        }});
    }}
}});"""

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
    app.run(host='0.0.0.0', port=5000, debug=True)