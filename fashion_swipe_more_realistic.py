# app.py - Fashion Swipe Studio Pro
import streamlit as st
import pandas as pd
import random
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import base64
import json
from collections import Counter
import gc
import os
import requests
from typing import List, Dict, Any, Optional, Tuple
import time

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Fashion Swipe Studio Pro",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #FF6B9D, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .fashion-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 20px 0;
        transition: all 0.4s ease-out;
    }
    
    .fashion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
    }
    
    .runway-preview {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    
    .professional-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #333;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 5px;
        display: inline-block;
    }
    
    .generation-alert {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #FFD700;
    }
    
    .api-status {
        background: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .api-status.unavailable {
        background: #fff5f5;
        border-left-color: #f56565;
    }
</style>
""", unsafe_allow_html=True)

# Fashion-MNIST Klassen
FASHION_CLASSES = {
    0: "T-Shirt/Top",
    1: "Hose", 
    2: "Pullover",
    3: "Kleid",
    4: "Mantel",
    5: "Sandalen",
    6: "Hemd",
    7: "Sneaker",
    8: "Tasche",
    9: "Stiefeletten"
}

class FashionGenerator:
    """Fashion Generator mit API-Integration"""
    
    def __init__(self):
        self.available_apis = self.check_api_availability()
        
    def check_api_availability(self) -> Dict[str, bool]:
        """Prüft verfügbare APIs"""
        apis = {
            'huggingface': bool(os.getenv('HUGGINGFACE_TOKEN')),
            'stability': bool(os.getenv('STABILITY_API_KEY')),
            'replicate': bool(os.getenv('REPLICATE_API_TOKEN')),
        }
        return apis
    
    def generate_with_huggingface_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt Hugging Face API mit schnellen Turbo-Modellen"""
        api_token = os.getenv('HUGGINGFACE_TOKEN')
        if not api_token:
            return None
        
        try:
            # Beste Modelle für Fashion-Fotografie (optimierte Reihenfolge)
            models = [
                "stabilityai/sdxl-turbo",           # 🚀 Speed: 1-4 Steps, ultra-schnell
                "SG161222/RealVisXL_V4.0",         # 📸 Fotorealismus: beste Menschen/Fashion
                "stabilityai/sd-turbo",             # ⚡ Alternative Turbo
                "SG161222/Realistic_Vision_V6.0_B1_noVAE",  # 📷 Realistisch, bewährt
                "runwayml/stable-diffusion-v1-5",   # 🔄 Zuverlässiger Klassiker
                "stabilityai/stable-diffusion-xl-base-1.0",  # 🎨 SDXL Basis
                "prompthero/openjourney-v4"         # 🎭 Künstlerischer Stil
            ]
            
            for model_idx, model in enumerate(models):
                try:
                    API_URL = f"https://api-inference.huggingface.co/models/{model}"
                    headers = {"Authorization": f"Bearer {api_token}"}
                    
                    # Model-spezifische Parameter für optimale Fashion-Ergebnisse
                    if "turbo" in model.lower():
                        # Turbo-Modelle: Sehr wenige Steps für Geschwindigkeit
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "negative_prompt": "blurry, bad quality, distorted, amateur, cartoon, anime, low resolution, deformed, ugly, bad anatomy",
                                "num_inference_steps": 2 if "sdxl-turbo" in model else 4,  
                                "guidance_scale": 0.0 if "sdxl-turbo" in model else 1.0,   
                                "width": 512,
                                "height": 768
                            }
                        }
                        model_name = "SDXL-Turbo ⚡" if "sdxl" in model else "SD-Turbo ⚡"
                        
                    elif "realvis" in model.lower() or "realistic_vision" in model.lower():
                        # RealVIS: Optimiert für Fotorealismus
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "negative_prompt": "blurry, bad quality, distorted, amateur, cartoon, anime, low resolution, deformed, ugly, bad anatomy, worst quality, low quality, normal quality, lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark)), (white letters)",
                                "num_inference_steps": 25,  # Mehr Steps für Realismus
                                "guidance_scale": 7.0,      # Optimiert für RealVIS
                                "width": 512,
                                "height": 768,
                                "scheduler": "DPMSolverMultistepScheduler"  # Besserer Scheduler für Realismus
                            }
                        }
                        model_name = "RealVIS 📸" if "realvis" in model.lower() else "Realistic Vision 📷"
                        
                    else:
                        # Standard-Modelle: Normale Parameter
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "negative_prompt": "blurry, bad quality, distorted, amateur, cartoon, anime, low resolution, deformed, ugly, bad anatomy",
                                "num_inference_steps": 20 if "xl" in model else 25,
                                "guidance_scale": 7.5,
                                "width": 512,
                                "height": 768
                            }
                        }
                        model_name = model.split('/')[-1].replace('-', ' ').replace('_', ' ').title()
                    
                    st.info(f"🎨 Versuche {model_name} (Model {model_idx + 1}/{len(models)})...")
                    
                    # Request mit Timeout
                    timeout = 30 if "turbo" in model.lower() else 60  # Turbo ist schneller
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'image' in content_type:
                            st.success(f"✅ {model_name} erfolgreich!")
                            return Image.open(io.BytesIO(response.content))
                        else:
                            # Manchmal JSON Response mit Error
                            try:
                                error_data = response.json()
                                if 'error' in error_data:
                                    st.warning(f"⚠️ {model_name}: {error_data['error'][:50]}...")
                                continue
                            except:
                                continue
                                
                    elif response.status_code == 503:
                        st.warning(f"⏳ {model_name} lädt... (ca. 20s)")
                        
                        # Bei Turbo-Modellen: Kurz warten und nochmal versuchen
                        if "turbo" in model.lower() and model_idx == 0:
                            st.info("⚡ Turbo-Model startet - 15 Sekunden warten...")
                            time.sleep(15)
                            
                            # Zweiter Versuch für Turbo
                            response2 = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                            if response2.status_code == 200:
                                content_type = response2.headers.get('content-type', '')
                                if 'image' in content_type:
                                    st.success(f"✅ {model_name} erfolgreich (2. Versuch)!")
                                    return Image.open(io.BytesIO(response2.content))
                        
                        continue
                        
                    elif response.status_code == 400:
                        st.warning(f"⚠️ {model_name}: Parameter-Fehler")
                        continue
                    elif response.status_code == 429:
                        st.warning(f"⚠️ {model_name}: Rate Limit - versuche nächstes Model")
                        continue
                    else:
                        st.warning(f"⚠️ {model_name}: HTTP {response.status_code}")
                        continue
                        
                except requests.Timeout:
                    st.warning(f"⏰ {model_name}: Timeout - versuche nächstes Model")
                    continue
                except Exception as e:
                    st.warning(f"❌ {model_name}: {str(e)[:50]}...")
                    continue
            
            # Alle Modelle fehlgeschlagen
            st.error("❌ Alle Hugging Face Modelle nicht verfügbar")
            return None
            
        except Exception as e:
            st.error(f"Hugging Face API Fehler: {e}")
            return None
    
    def create_fallback_design(self, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Erstellt Fallback-Design"""
        
        # Canvas
        img = Image.new('RGB', (512, 768), color=(245, 245, 250))
        
        # Hintergrund
        img = self.create_background(img, style_prompt)
        
        # Model
        model_img = self.create_model(img.size)
        
        # Styling
        model_img = self.apply_styling(model_img, selected_items, style_prompt)
        
        # Combine
        final_img = Image.alpha_composite(img.convert('RGBA'), model_img.convert('RGBA'))
        
        # Enhance
        final_img = self.enhance_image(final_img.convert('RGB'))
        
        return final_img
    
    def create_background(self, base_img: Image.Image, style_prompt: str) -> Image.Image:
        """Erstellt styled Hintergrund"""
        draw = ImageDraw.Draw(base_img)
        width, height = base_img.size
        
        # Style-basierte Hintergründe
        if "vintage" in style_prompt.lower():
            # Vintage sepia
            for y in range(height):
                shade = int(235 + (y / height) * 20)
                draw.rectangle([0, y, width, y+1], fill=(shade, shade-15, shade-30))
        elif "modern" in style_prompt.lower():
            # Modern gradient
            for y in range(height):
                shade = int(250 - (y / height) * 30)
                draw.rectangle([0, y, width, y+1], fill=(shade, shade, shade+5))
        else:
            # Standard
            for y in range(height):
                shade = int(240 + (y / height) * 15)
                draw.rectangle([0, y, width, y+1], fill=(shade, shade, shade+5))
        
        # Beleuchtung
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
        light_draw = ImageDraw.Draw(overlay)
        
        center_x, center_y = width // 2, height // 4
        for radius in range(200, 0, -10):
            alpha = int(12 * (200 - radius) / 200)
            light_draw.ellipse([
                center_x - radius, center_y - radius//2,
                center_x + radius, center_y + radius//2
            ], fill=(255, 255, 255, alpha))
        
        base_img = Image.alpha_composite(base_img.convert('RGBA'), overlay).convert('RGB')
        return base_img
    
    def create_model(self, size: Tuple[int, int]) -> Image.Image:
        """Erstellt Model-Silhouette"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center_x = width // 2
        skin_color = (245, 220, 177, 255)
        
        # Proportionen
        head_y = height // 8
        shoulder_y = head_y + 60
        waist_y = shoulder_y + 120
        hip_y = waist_y + 80
        knee_y = hip_y + 140
        ankle_y = height - 80
        
        # Kopf
        draw.ellipse([center_x - 30, head_y - 30, center_x + 30, head_y + 30], fill=skin_color)
        
        # Gesicht
        draw.ellipse([center_x - 12, head_y - 5, center_x - 8, head_y + 5], fill=(70, 50, 30, 255))
        draw.ellipse([center_x + 8, head_y - 5, center_x + 12, head_y + 5], fill=(70, 50, 30, 255))
        
        # Haare
        hair_color = (101, 67, 33, 255)
        draw.ellipse([center_x - 35, head_y - 35, center_x + 35, head_y + 10], fill=hair_color)
        
        # Körper
        body_points = [
            (center_x - 40, shoulder_y),
            (center_x + 40, shoulder_y),
            (center_x + 30, waist_y),
            (center_x + 35, hip_y),
            (center_x - 35, hip_y),
            (center_x - 30, waist_y)
        ]
        draw.polygon(body_points, fill=skin_color)
        
        # Arme
        draw.polygon([
            (center_x - 40, shoulder_y),
            (center_x - 60, shoulder_y + 40),
            (center_x - 55, waist_y - 20),
            (center_x - 40, waist_y - 30)
        ], fill=skin_color)
        
        draw.polygon([
            (center_x + 40, shoulder_y),
            (center_x + 60, shoulder_y + 40),
            (center_x + 55, waist_y - 20),
            (center_x + 40, waist_y - 30)
        ], fill=skin_color)
        
        # Beine
        draw.polygon([
            (center_x - 35, hip_y),
            (center_x - 15, hip_y),
            (center_x - 15, knee_y),
            (center_x - 20, ankle_y),
            (center_x - 35, ankle_y),
            (center_x - 30, knee_y)
        ], fill=skin_color)
        
        draw.polygon([
            (center_x + 15, hip_y),
            (center_x + 35, hip_y),
            (center_x + 30, knee_y),
            (center_x + 35, ankle_y),
            (center_x + 20, ankle_y),
            (center_x + 15, knee_y)
        ], fill=skin_color)
        
        return img
    
    def apply_styling(self, model_img: Image.Image, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Wendet Styling an"""
        draw = ImageDraw.Draw(model_img)
        width, height = model_img.size
        center_x = width // 2
        
        colors = self.get_colors(style_prompt)
        categories = [item['category'] for item in selected_items]
        
        for i, category in enumerate(set(categories)):
            color = colors[i % len(colors)] + (255,)
            
            if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
                # Oberteil
                draw.polygon([
                    (center_x - 35, 80),
                    (center_x + 35, 80),
                    (center_x + 25, 200),
                    (center_x - 25, 200)
                ], fill=color, outline=self.darken_color(color[:3]) + (255,), width=2)
                
            elif category == "Kleid":
                # Kleid
                draw.polygon([
                    (center_x - 35, 80),
                    (center_x + 35, 80),
                    (center_x + 25, 200),
                    (center_x + 40, 380),
                    (center_x - 40, 380),
                    (center_x - 25, 200)
                ], fill=color, outline=self.darken_color(color[:3]) + (255,), width=2)
                
            elif category == "Hose":
                # Hose
                draw.polygon([
                    (center_x - 25, 200),
                    (center_x - 5, 200),
                    (center_x - 10, 480),
                    (center_x - 30, 480)
                ], fill=color, outline=self.darken_color(color[:3]) + (255,), width=2)
                
                draw.polygon([
                    (center_x + 5, 200),
                    (center_x + 25, 200),
                    (center_x + 30, 480),
                    (center_x + 10, 480)
                ], fill=color, outline=self.darken_color(color[:3]) + (255,), width=2)
                
            elif category == "Mantel":
                # Mantel
                draw.polygon([
                    (center_x - 45, 70),
                    (center_x + 45, 70),
                    (center_x + 35, 200),
                    (center_x + 50, 400),
                    (center_x - 50, 400),
                    (center_x - 35, 200)
                ], fill=color, outline=self.darken_color(color[:3]) + (255,), width=3)
        
        return model_img
    
    def enhance_image(self, img: Image.Image) -> Image.Image:
        """Verbessert Bildqualität"""
        # Schärfe
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        # Kontrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.05)
        
        return img
    
    def get_colors(self, style_prompt: str) -> List[Tuple[int, int, int]]:
        """Gibt Farben basierend auf Style zurück"""
        style_lower = style_prompt.lower()
        
        if "modern" in style_lower:
            return [(45, 45, 48), (240, 240, 245), (128, 128, 130)]
        elif "klassisch" in style_lower:
            return [(25, 25, 112), (255, 255, 255), (139, 69, 19)]
        elif "vintage" in style_lower:
            return [(139, 69, 19), (205, 133, 63), (128, 0, 0)]
        elif "sportlich" in style_lower:
            return [(0, 0, 0), (255, 69, 0), (255, 255, 255)]
        else:
            return [(47, 79, 79), (205, 192, 176), (139, 69, 19)]
    
    def darken_color(self, color: Tuple[int, int, int], factor: float = 0.7) -> Tuple[int, int, int]:
        """Dunkelt Farbe ab"""
        return tuple(max(0, int(c * factor)) for c in color[:3])

def create_fashion_prompt(selected_items: List[Dict], style_prompt: str) -> str:
    """Erstellt Fashion-Prompt"""
    categories = [item['category'] for item in selected_items]
    category_text = ", ".join(set(categories))
    
    base_prompt = f"professional fashion photography, elegant female model wearing {category_text}"
    
    style_additions = {
        "modern": "contemporary minimalist design, clean lines, urban chic",
        "klassisch": "timeless elegant style, refined sophisticated look",
        "vintage": "retro inspired fashion, classic vintage aesthetic",
        "sportlich": "athletic wear, dynamic pose, fitness lifestyle",
        "extravagant": "high fashion couture, dramatic styling, avant-garde",
        "minimalistisch": "clean minimal aesthetic, neutral colors, simple elegance"
    }
    
    style_addition = ""
    for key, addition in style_additions.items():
        if key in style_prompt.lower():
            style_addition = addition
            break
    
    if not style_addition:
        style_addition = "contemporary fashion styling, modern elegance"
    
    full_prompt = f"{base_prompt}, {style_addition}, professional studio lighting, high fashion photography, photorealistic, detailed, 8k quality"
    
    return full_prompt

def create_sample_fashion_items():
    """Lädt echte Fashion-MNIST Daten"""
    try:
        # Versuche Fashion-MNIST zu laden
        import tensorflow as tf
        
        # Lade Fashion-MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Nimm eine kleinere Auswahl für bessere Performance
        num_items = 20
        random_indices = np.random.choice(len(x_train), size=num_items, replace=False)
        
        items = []
        brands = ["Chanel", "Dior", "Versace", "Prada", "Gucci", "Armani", "Zara", "H&M", "COS", "Uniqlo"]
        prices = ["45€", "65€", "85€", "120€", "150€", "200€", "250€", "300€", "450€", "600€"]
        
        for i, idx in enumerate(random_indices):
            image = x_train[idx]
            label = y_train[idx]
            category = FASHION_CLASSES[label]
            
            item = {
                "id": i + 1,
                "name": f"{category} Collection #{idx}",
                "brand": random.choice(brands),
                "price": random.choice(prices),
                "category": category,
                "description": f"Premium {category.lower()} aus der neuesten Kollektion",
                "image_data": image_to_base64(image),
                "original_index": idx,
                "label": label,
                "timestamp": datetime.now().isoformat()
            }
            items.append(item)
        
        # Cleanup für Memory
        del x_train, y_train, x_test, y_test
        import gc
        gc.collect()
        
        return items
        
    except ImportError:
        # Fallback: TensorFlow nicht verfügbar, verwende bessere Sample-Bilder
        st.warning("📦 TensorFlow nicht verfügbar - verwende Sample-Daten")
        return create_enhanced_sample_items()
    except Exception as e:
        st.warning(f"⚠️ Fashion-MNIST Fehler: {e} - verwende Sample-Daten")
        return create_enhanced_sample_items()

def create_enhanced_sample_items():
    """Erstellt verbesserte Sample Fashion Items (falls Fashion-MNIST nicht verfügbar)"""
    items = []
    brands = ["Chanel", "Dior", "Versace", "Prada", "Gucci", "Armani", "Zara", "H&M", "COS", "Uniqlo"]
    prices = ["45€", "65€", "85€", "120€", "150€", "200€", "250€", "300€", "450€", "600€"]
    
    for i in range(20):
        category_idx = i % len(FASHION_CLASSES)
        category = FASHION_CLASSES[category_idx]
        
        # Erstelle realistischere Fashion-Bilder
        img = Image.new('RGB', (28, 28), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        
        # Verschiedene realistische Farben
        colors = [
            (50, 50, 50),      # Schwarz
            (100, 50, 50),     # Dunkelrot
            (50, 50, 100),     # Navy
            (100, 80, 60),     # Braun
            (80, 80, 80),      # Grau
            (120, 100, 80),    # Beige
            (60, 80, 60),      # Olivgrün
            (80, 60, 80),      # Violett
            (100, 100, 60),    # Senfgelb
            (60, 100, 100),    # Türkis
        ]
        
        base_color = colors[i % len(colors)]
        
        # Realistischere Formen je nach Kategorie
        if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
            # Oberteil mit realistischer Form
            draw.polygon([(6, 8), (22, 8), (24, 12), (20, 20), (8, 20), (4, 12)], fill=base_color)
            # Ärmel
            draw.rectangle([2, 10, 6, 18], fill=base_color)
            draw.rectangle([22, 10, 26, 18], fill=base_color)
            # Details
            draw.line([(10, 8), (18, 8)], fill=tuple(c+30 for c in base_color), width=1)
            
        elif category == "Kleid":
            # Kleid mit A-Linie
            draw.polygon([(8, 6), (20, 6), (22, 12), (26, 26), (2, 26), (6, 12)], fill=base_color)
            # Ärmel
            draw.rectangle([4, 8, 8, 14], fill=base_color)
            draw.rectangle([20, 8, 24, 14], fill=base_color)
            # Taille
            draw.line([(6, 16), (22, 16)], fill=tuple(c+20 for c in base_color), width=1)
            
        elif category == "Hose":
            # Realistische Hose
            draw.rectangle([8, 12, 12, 26], fill=base_color)
            draw.rectangle([16, 12, 20, 26], fill=base_color)
            # Bund
            draw.rectangle([6, 10, 22, 14], fill=tuple(c+15 for c in base_color))
            # Bügelfalte
            draw.line([(10, 14), (10, 26)], fill=tuple(c+25 for c in base_color), width=1)
            draw.line([(18, 14), (18, 26)], fill=tuple(c+25 for c in base_color), width=1)
            
        elif category == "Mantel":
            # Langer Mantel
            draw.polygon([(4, 6), (24, 6), (26, 12), (24, 26), (4, 26), (2, 12)], fill=base_color)
            # Ärmel
            draw.rectangle([0, 8, 4, 22], fill=base_color)
            draw.rectangle([24, 8, 28, 22], fill=base_color)
            # Knöpfe
            for y in [10, 14, 18, 22]:
                draw.ellipse([13, y, 15, y+2], fill=tuple(c+40 for c in base_color))
            # Kragen
            draw.polygon([(10, 6), (18, 6), (20, 10), (8, 10)], fill=tuple(c+20 for c in base_color))
            
        elif category in ["Sneaker", "Sandalen", "Stiefeletten"]:
            if category == "Stiefeletten":
                # Hohe Stiefel
                draw.ellipse([4, 16, 12, 24], fill=base_color)
                draw.ellipse([16, 16, 24, 24], fill=base_color)
                draw.rectangle([5, 8, 11, 18], fill=base_color)
                draw.rectangle([17, 8, 23, 18], fill=base_color)
                # Schnürung
                for y in [10, 13, 16]:
                    draw.line([(6, y), (10, y)], fill=tuple(c+30 for c in base_color), width=1)
                    draw.line([(18, y), (22, y)], fill=tuple(c+30 for c in base_color), width=1)
            else:
                # Normale Schuhe
                draw.ellipse([4, 18, 12, 26], fill=base_color)
                draw.ellipse([16, 18, 24, 26], fill=base_color)
                # Sohle
                draw.arc([4, 20, 12, 26], 0, 180, fill=tuple(c+40 for c in base_color), width=2)
                draw.arc([16, 20, 24, 26], 0, 180, fill=tuple(c+40 for c in base_color), width=2)
                
        elif category == "Tasche":
            # Handtasche
            draw.rectangle([8, 12, 20, 22], fill=base_color)
            # Henkel
            draw.arc([10, 8, 18, 14], 0, 180, fill=tuple(c+20 for c in base_color), width=2)
            # Verschluss
            draw.rectangle([12, 12, 16, 14], fill=tuple(c+30 for c in base_color))
            # Details
            draw.rectangle([9, 16, 19, 18], fill=tuple(c+15 for c in base_color))
            
        else:  # Sandalen
            # Sandalen
            draw.ellipse([4, 20, 12, 26], fill=base_color)
            draw.ellipse([16, 20, 24, 26], fill=base_color)
            # Riemen
            draw.line([(6, 18), (10, 22)], fill=tuple(c+25 for c in base_color), width=2)
            draw.line([(18, 18), (22, 22)], fill=tuple(c+25 for c in base_color), width=2)
        
        # Schatten für Tiefe
        shadow_img = Image.new('RGBA', (28, 28), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        
        # Leichter Schatten rechts unten
        if category != "Tasche":
            shadow_draw.ellipse([12, 22, 20, 27], fill=(0, 0, 0, 30))
        
        img = Image.alpha_composite(img.convert('RGBA'), shadow_img).convert('RGB')
        
        item = {
            "id": i + 1,
            "name": f"{category} Collection #{i+1}",
            "brand": random.choice(brands),
            "price": random.choice(prices),
            "category": category,
            "description": f"Premium {category.lower()} aus der neuesten Kollektion",
            "image_data": image_to_base64(np.array(img)),
            "original_index": i,
            "label": category_idx,
            "timestamp": datetime.now().isoformat()
        }
        items.append(item)
    
    return items

def image_to_base64(image_array):
    """Konvertiert Bild zu Base64"""
    if len(image_array.shape) == 2:
        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    else:
        image = Image.fromarray(image_array.astype(np.uint8))
    
    image = image.resize((280, 280), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def numpy_to_base64(image_array, size=(512, 768)):
    """Konvertiert NumPy Array zu Base64"""
    if len(image_array.shape) == 2:
        image = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
    else:
        image = Image.fromarray((image_array * 255).astype(np.uint8))
    
    if size:
        image = image.resize(size, Image.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def generate_fashion_design(selected_items: List[Dict], style_prompt: str, 
                          generator: FashionGenerator) -> Optional[np.ndarray]:
    """Generiert Fashion-Design"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🎨 Erstelle Fashion-Prompt...")
        progress_bar.progress(0.1)
        
        prompt = create_fashion_prompt(selected_items, style_prompt)
        
        st.write(f"**🎭 Generierungs-Prompt:** {prompt[:120]}...")
        
        available_apis = generator.check_api_availability()
        generated_image = None
        
        # Versuche Hugging Face API
        if available_apis['huggingface']:
            status_text.text("🤗 Generiere mit Hugging Face API...")
            progress_bar.progress(0.4)
            generated_image = generator.generate_with_huggingface_api(prompt)
            if generated_image:
                status_text.text("✅ Hugging Face erfolgreich!")
        
        if generated_image is None:
            # Fallback
            status_text.text("🎨 Erstelle optimiertes Fallback-Design...")
            progress_bar.progress(0.8)
            generated_image = generator.create_fallback_design(selected_items, style_prompt)
            status_text.text("✅ Fallback-Design erstellt!")
        
        progress_bar.progress(1.0)
        
        return np.array(generated_image).astype(np.float32) / 255.0
        
    except Exception as e:
        st.error(f"❌ Fehler bei der Generierung: {e}")
        try:
            fallback_image = generator.create_fallback_design(selected_items, style_prompt)
            return np.array(fallback_image).astype(np.float32) / 255.0
        except Exception:
            return None

# Session State
def init_session_state():
    """Initialisiert Session State"""
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'liked_items' not in st.session_state:
        st.session_state.liked_items = []
    if 'disliked_items' not in st.session_state:
        st.session_state.disliked_items = []
    if 'fashion_items' not in st.session_state:
        st.session_state.fashion_items = create_sample_fashion_items()
    if 'all_time_favorites' not in st.session_state:
        st.session_state.all_time_favorites = []
    if 'selected_for_generation' not in st.session_state:
        st.session_state.selected_for_generation = []
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'generator' not in st.session_state:
        st.session_state.generator = FashionGenerator()

def like_item():
    """Liked ein Item"""
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        current_item = items[st.session_state.current_index]
        st.session_state.liked_items.append(current_item)
        
        if not any(fav['original_index'] == current_item['original_index'] 
                  for fav in st.session_state.all_time_favorites):
            st.session_state.all_time_favorites.append(current_item)
        
        st.session_state.current_index += 1

def dislike_item():
    """Disliked ein Item"""
    if st.session_state.current_index < len(st.session_state.fashion_items):
        st.session_state.current_index += 1

def reset_session():
    """Setzt Session zurück"""
    st.session_state.current_index = 0
    st.session_state.liked_items = []
    st.session_state.disliked_items = []
    st.session_state.fashion_items = create_sample_fashion_items()

def render_swipe_tab():
    """Rendert den Swipe-Tab"""
    items = st.session_state.fashion_items
    current_idx = st.session_state.current_index
    total_items = len(items)
    
    # Progress Bar
    progress = current_idx / total_items if total_items > 0 else 0
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #FF6B9D {progress * 100}%, #f0f0f0 {progress * 100}%); 
                height: 12px; border-radius: 6px; margin: 20px 0;"></div>
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
        <strong>{current_idx}</strong> von <strong>{total_items}</strong> Fashion-Artikeln durchgesehen
    </p>
    """, unsafe_allow_html=True)
    
    if current_idx >= total_items:
        st.success(f"🎉 Session beendet! {total_items} Artikel durchgesehen.")
        st.markdown(f"**Favoriten gesammelt:** {len(st.session_state.liked_items)}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Neue Fashion-Session starten", type="primary", use_container_width=True):
                reset_session()
                st.rerun()
    else:
        current_item = items[current_idx]
        
        # Hauptkarte
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col2:
            st.markdown(f"""
            <div class="fashion-card">
                <div style="text-align: center;">
                    <h2 style="color: #333; margin-bottom: 10px;">{current_item['category']}</h2>
                    <div class="professional-badge">{current_item['brand']}</div>
                    <div class="professional-badge">{current_item['price']}</div>
                    <div style="margin: 20px 0;">
                        <img src="{current_item['image_data']}" 
                             style="width: 100%; max-width: 350px; border-radius: 15px; 
                                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    </div>
                    <p style="color: #666; font-style: italic; margin-top: 15px;">
                        {current_item['description']}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("👎 Skip", type="secondary", use_container_width=True):
                dislike_item()
                st.rerun()
        
        with col3:
            if st.button("❤️ Love it!", type="primary", use_container_width=True):
                like_item()
                st.rerun()

def render_generate_tab():
    """Rendert den Generate-Tab"""
    st.markdown("## 🎨 Fashion-Generator Pro")
    
    # API Status anzeigen
    generator = st.session_state.generator
    available_apis = generator.check_api_availability()
    
    if available_apis['huggingface']:
        st.markdown("""
        <div class="generation-alert">
            <h3>🚀 Hugging Face API aktiv!</h3>
            <p>✅ <strong>Stable Diffusion XL</strong> verfügbar<br>
            ✅ <strong>Professionelle Fashion-Fotografien</strong><br>
            ✅ <strong>Kostenlose Premium-Qualität</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="generation-alert">
            <h3>💡 Hugging Face Setup</h3>
            <p>Für AI-generierte Fashion-Fotos, füge deinen Token in den Streamlit Secrets hinzu:</p>
            <pre>HUGGINGFACE_TOKEN = "hf_dein_token_hier"</pre>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.all_time_favorites:
        st.warning("💡 Sammle erst Favoriten durch Swipen!")
        return
    
    # Favoriten-Auswahl
    st.markdown("### 👗 Wähle Fashion-Styles")
    
    cols = st.columns(5)
    for idx, item in enumerate(st.session_state.all_time_favorites[:15]):
        with cols[idx % 5]:
            is_selected = item in st.session_state.selected_for_generation
            
            if st.button(
                "✅" if is_selected else "⭕",
                key=f"sel_{idx}",
                help=f"{item['category']} - {item['brand']}",
                use_container_width=True
            ):
                if is_selected:
                    st.session_state.selected_for_generation.remove(item)
                else:
                    if len(st.session_state.selected_for_generation) < 3:
                        st.session_state.selected_for_generation.append(item)
                    else:
                        st.warning("⚠️ Maximal 3 Styles!")
                st.rerun()
            
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="{item['image_data']}" 
                     style="width: 100%; border-radius: 8px; margin-bottom: 5px;
                            border: {'3px solid #4CAF50' if is_selected else '1px solid #ddd'};">
                <div style="font-size: 0.8rem; color: #666;">{item['category']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.selected_for_generation:
        st.markdown(f"""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <strong>✨ Ausgewählt:</strong> {len(st.session_state.selected_for_generation)} Styles<br>
            <em>{', '.join([item['category'] for item in st.session_state.selected_for_generation])}</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Style-Konfiguration
        col1, col2 = st.columns(2)
        
        with col1:
            style_mood = st.selectbox("🎭 Style", ["Modern", "Klassisch", "Vintage", "Sportlich", "Minimalistisch", "Extravagant"])
        
        with col2:
            color_scheme = st.selectbox("🎨 Farben", ["Natürlich", "Monochrom", "Pastell", "Kräftig"])
        
        style_prompt = f"{style_mood.lower()} style, {color_scheme.lower()} colors"
        
        # Generate-Button
        st.markdown("---")
        if st.button("🎨 ✨ Generiere Fashion-Design!", type="primary", use_container_width=True):
            with st.spinner("Erstelle Fashion-Design..."):
                generated_image = generate_fashion_design(
                    st.session_state.selected_for_generation,
                    style_prompt,
                    st.session_state.generator
                )
                
                if generated_image is not None:
                    # Speichere Generierung
                    api_used = "Hugging Face AI" if available_apis['huggingface'] else "Enhanced Local"
                    generation_data = {
                        'image': numpy_to_base64(generated_image, size=(512, 768)),
                        'style': style_mood,
                        'colors': color_scheme,
                        'items': [item['category'] for item in st.session_state.selected_for_generation],
                        'timestamp': datetime.now().isoformat(),
                        'api_used': api_used
                    }
                    
                    st.session_state.generated_images.append(generation_data)
                    if len(st.session_state.generated_images) > 15:
                        st.session_state.generated_images.pop(0)
                    
                    # Darstellung
                    st.markdown("---")
                    st.markdown("## 📸 Generiertes Fashion-Design")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("""
                        <div class="runway-preview">
                            <div style="position: relative; z-index: 1;">
                        """, unsafe_allow_html=True)
                        
                        st.image(generated_image, 
                               caption=f"Fashion-Design ({api_used})",
                               use_column_width=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: white; padding: 20px; border-radius: 15px; 
                                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                            <h4>📋 Design-Details</h4>
                            <p><strong>Style:</strong> {style_mood}</p>
                            <p><strong>Farben:</strong> {color_scheme}</p>
                            <p><strong>Items:</strong> {', '.join([item['category'] for item in st.session_state.selected_for_generation])}</p>
                            <p><strong>Generiert mit:</strong> {api_used}</p>
                            <p><strong>Zeit:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download-Button
                        img_buffer = io.BytesIO()
                        img = Image.fromarray((generated_image * 255).astype(np.uint8))
                        img.save(img_buffer, format="PNG", optimize=True, quality=90)
                        img_buffer.seek(0)
                        
                        st.download_button(
                            "💾 Design herunterladen",
                            data=img_buffer,
                            file_name=f"fashion_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                else:
                    st.error("❌ Fehler bei der Generierung. Bitte versuche es erneut.")

def render_gallery_tab():
    """Zeigt Galerie"""
    st.markdown("## 🖼️ Design-Galerie")
    
    if not st.session_state.generated_images:
        st.markdown("""
        <div style="text-align: center; padding: 60px; color: #666;">
            <h3>📷 Noch keine Designs generiert</h3>
            <p>Gehe zum Generator-Tab und erstelle dein erstes Fashion-Design!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"**📸 {len(st.session_state.generated_images)} Fashion-Designs generiert**")
    
    # Galerie
    for idx, gen_data in enumerate(reversed(st.session_state.generated_images)):
        with st.container():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.image(gen_data['image'], use_column_width=True)
            
            with col2:
                api_badge = "🤖 AI" if "AI" in gen_data.get('api_used', '') else "🎨 Local"
                
                st.markdown(f"""
                <div style="padding: 20px;">
                    <h4>📸 Design #{len(st.session_state.generated_images) - idx}</h4>
                    <div style="background: #f8f9fa; padding: 8px; border-radius: 5px; margin: 8px 0;">
                        <strong>{api_badge} {gen_data.get('api_used', 'N/A')}</strong>
                    </div>
                    <p><strong>Style:</strong> {gen_data.get('style', 'N/A')}</p>
                    <p><strong>Farben:</strong> {gen_data.get('colors', 'N/A')}</p>
                    <p><strong>Items:</strong> {', '.join(gen_data.get('items', []))}</p>
                    <p><strong>Erstellt:</strong> {gen_data.get('timestamp', 'N/A')[:19].replace('T', ' ')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")

def main():
    """Hauptfunktion"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">🏃‍♀️ Fashion Swipe Studio</div>
    <div class="sub-header">Funktioniert mit Hugging Face AI!</div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔄 Fashion Swipe", "🎨 AI Generator", "🖼️ Galerie"])
    
    with tab1:
        render_swipe_tab()
    
    with tab2:
        render_generate_tab()
    
    with tab3:
        render_gallery_tab()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Statistiken")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("❤️ Likes", len(st.session_state.liked_items))
        with col2:
            st.metric("⭐ Favoriten", len(st.session_state.all_time_favorites))
        
        st.metric("📸 Designs", len(st.session_state.generated_images))
        
        st.markdown("---")
        
        # Action Buttons
        if st.button("🔄 Reset", use_container_width=True):
            reset_session()
            st.rerun()
        
        if st.button("🗑️ Galerie leeren", use_container_width=True):
            st.session_state.generated_images = []
            st.rerun()
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔧 API Status")
        generator = st.session_state.generator
        available_apis = generator.check_api_availability()
        
        if available_apis['huggingface']:
            st.markdown("""
            <div class="api-status">
                <strong>✅ Hugging Face</strong><br>
                <small>Token aktiv</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="api-status unavailable">
                <strong>❌ Hugging Face</strong><br>
                <small>Token fehlt</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; font-size: 0.8rem;">
            <strong>🚀 Fashion Swipe Studio:</strong><br>
            • Hugging Face AI Integration<br>
            • Stable Diffusion XL<br>
            • Professionelle Fashion-Fotos<br>
            • Kostenlose Premium-Qualität<br>
            • Optimiert für Streamlit Cloud
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
