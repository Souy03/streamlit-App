import streamlit as st
import pandas as pd
import random
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import io
import base64
import json
from collections import Counter
import gc
import os
import requests
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Fashion Swipe Studio",
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

class CloudOptimizedFashionGenerator:
    """Optimiert für Streamlit Cloud - keine schweren AI-Modelle"""
    
    def __init__(self):
        self.available_apis = self.check_api_availability()
        
    def check_api_availability(self) -> Dict[str, bool]:
        """Prüft verfügbare APIs"""
        apis = {
            'huggingface': bool(os.getenv('HUGGINGFACE_TOKEN')),
            'replicate': bool(os.getenv('REPLICATE_API_TOKEN')),
            'stability': bool(os.getenv('STABILITY_API_KEY'))
        }
        return apis
    
    def generate_with_stability_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt Stability AI API (am zuverlässigsten für Cloud)"""
        api_key = os.getenv('STABILITY_API_KEY')
        if not api_key:
            return None
            
        try:
            import requests
            
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            
            body = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1
                    },
                    {
                        "text": "blurry, bad quality, distorted, amateur, low resolution",
                        "weight": -1
                    }
                ],
                "cfg_scale": 7,
                "height": 768,
                "width": 512,
                "samples": 1,
                "steps": 30,
                "style_preset": "photographic"
            }
            
            response = requests.post(url, headers=headers, json=body, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("artifacts"):
                    image_data = data["artifacts"][0]["base64"]
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    return image
            else:
                st.error(f"Stability API Error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Stability API Fehler: {e}")
            return None
    
    def generate_with_replicate_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt Replicate API als Alternative"""
        api_token = os.getenv('REPLICATE_API_TOKEN')
        if not api_token:
            return None
            
        try:
            headers = {
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json"
            }
            
            # Verwende SDXL über Replicate
            data = {
                "version": "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                "input": {
                    "prompt": prompt,
                    "negative_prompt": "blurry, bad quality, distorted, amateur",
                    "width": 512,
                    "height": 768,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 25
                }
            }
            
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 201:
                prediction_url = response.json()["urls"]["get"]
                
                # Warte auf Ergebnis (mit Timeout)
                for _ in range(30):  # Max 30 Sekunden warten
                    result_response = requests.get(prediction_url, headers=headers, timeout=5)
                    result = result_response.json()
                    
                    if result["status"] == "succeeded" and result.get("output"):
                        image_url = result["output"][0]
                        img_response = requests.get(image_url, timeout=10)
                        if img_response.status_code == 200:
                            return Image.open(io.BytesIO(img_response.content))
                    elif result["status"] == "failed":
                        break
                    
                    time.sleep(1)
            
            return None
            
        except Exception as e:
            st.error(f"Replicate API Fehler: {e}")
            return None
    
    def generate_with_huggingface_inference(self, prompt: str) -> Optional[Image.Image]:
        """Verbesserte Hugging Face API Nutzung"""
        api_token = os.getenv('HUGGINGFACE_TOKEN')
        if not api_token:
            return None
        
        try:
            # Verwende verschiedene Modelle
            models = [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4"
            ]
            
            for model in models:
                try:
                    API_URL = f"https://api-inference.huggingface.co/models/{model}"
                    headers = {"Authorization": f"Bearer {api_token}"}
                    
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "negative_prompt": "blurry, bad quality, distorted",
                            "num_inference_steps": 20,
                            "guidance_scale": 7.5,
                            "width": 512,
                            "height": 768
                        }
                    }
                    
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                    
                    if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
                        return Image.open(io.BytesIO(response.content))
                    elif response.status_code == 503:
                        # Model loading - try next model
                        continue
                        
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            st.warning(f"Hugging Face API nicht verfügbar: {e}")
            return None
    
    def create_professional_fashion_design(self, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Erstellt hochwertiges Fashion-Design ohne AI-Abhängigkeit"""
        
        # Hochauflösender Canvas
        img = Image.new('RGB', (512, 768), color=(245, 245, 250))
        
        # Professioneller Hintergrund
        img = self.create_studio_background(img)
        
        # Realistisches Model
        model_img = self.create_detailed_model(img.size)
        
        # Styling basierend auf Auswahl
        model_img = self.apply_fashion_styling(model_img, selected_items, style_prompt)
        
        # Professionelle Effekte
        model_img = self.add_professional_effects(model_img)
        
        # Composite
        final_img = Image.alpha_composite(img.convert('RGBA'), model_img.convert('RGBA'))
        
        return final_img.convert('RGB')
    
    def create_studio_background(self, base_img: Image.Image) -> Image.Image:
        """Erstellt professionellen Studio-Hintergrund"""
        draw = ImageDraw.Draw(base_img)
        width, height = base_img.size
        
        # Studio-Gradient
        for y in range(height):
            shade = int(240 + (y / height) * 15)
            draw.rectangle([0, y, width, y+1], fill=(shade, shade, shade+5))
        
        # Studio-Beleuchtung simulieren
        # Hauptlicht
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
        light_draw = ImageDraw.Draw(overlay)
        
        center_x, center_y = width // 2, height // 4
        for radius in range(200, 0, -10):
            alpha = int(15 * (200 - radius) / 200)
            light_draw.ellipse([
                center_x - radius, center_y - radius//2,
                center_x + radius, center_y + radius//2
            ], fill=(255, 255, 255, alpha))
        
        base_img.paste(overlay, (0, 0), overlay)
        return base_img
    
    def create_detailed_model(self, size: Tuple[int, int]) -> Image.Image:
        """Erstellt detailliertes Model"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center_x = width // 2
        
        # Proportionen
        head_y = height // 8
        neck_y = head_y + 50
        shoulder_y = neck_y + 30
        chest_y = shoulder_y + 80
        waist_y = chest_y + 100
        hip_y = waist_y + 80
        knee_y = hip_y + 150
        ankle_y = height - 60
        
        # Hautfarbe
        skin_color = (245, 220, 177)
        
        # Kopf (detaillierter)
        head_width, head_height = 50, 65
        draw.ellipse([
            center_x - head_width//2, head_y - head_height//2,
            center_x + head_width//2, head_y + head_height//2
        ], fill=skin_color, outline=(200, 180, 140), width=2)
        
        # Gesichtszüge
        eye_y = head_y - 8
        draw.ellipse([center_x - 18, eye_y - 3, center_x - 12, eye_y + 3], fill=(70, 50, 30))
        draw.ellipse([center_x + 12, eye_y - 3, center_x + 18, eye_y + 3], fill=(70, 50, 30))
        draw.ellipse([center_x - 3, head_y + 5, center_x + 3, head_y + 12], fill=(220, 180, 140))
        draw.arc([center_x - 8, head_y + 18, center_x + 8, head_y + 25], 0, 180, fill=(200, 120, 120), width=2)
        
        # Haare
        hair_color = (101, 67, 33)
        hair_points = [
            (center_x - head_width//2 - 8, head_y - head_height//2 - 15),
            (center_x + head_width//2 + 8, head_y - head_height//2 - 15),
            (center_x + head_width//2 + 5, head_y + head_height//2 - 10),
            (center_x - head_width//2 - 5, head_y + head_height//2 - 10)
        ]
        draw.polygon(hair_points, fill=hair_color)
        
        # Hals
        draw.rectangle([center_x - 12, neck_y, center_x + 12, shoulder_y], fill=skin_color)
        
        # Körper (realistischer)
        shoulder_width = 65
        chest_width = 60
        waist_width = 45
        hip_width = 65
        
        # Torso
        torso_points = [
            (center_x - shoulder_width//2, shoulder_y),
            (center_x + shoulder_width//2, shoulder_y),
            (center_x + chest_width//2, chest_y),
            (center_x + waist_width//2, waist_y),
            (center_x + hip_width//2, hip_y),
            (center_x - hip_width//2, hip_y),
            (center_x - waist_width//2, waist_y),
            (center_x - chest_width//2, chest_y)
        ]
        draw.polygon(torso_points, fill=skin_color, outline=(200, 180, 140), width=1)
        
        # Arme
        arm_width = 20
        # Linker Arm
        left_arm_points = [
            (center_x - shoulder_width//2, shoulder_y),
            (center_x - shoulder_width//2 - 35, shoulder_y + 80),
            (center_x - shoulder_width//2 - 30, waist_y),
            (center_x - shoulder_width//2 - 10, waist_y),
            (center_x - shoulder_width//2 - 15, shoulder_y + 80),
            (center_x - shoulder_width//2 - arm_width, shoulder_y)
        ]
        draw.polygon(left_arm_points, fill=skin_color, outline=(200, 180, 140), width=1)
        
        # Rechter Arm
        right_arm_points = [
            (center_x + shoulder_width//2, shoulder_y),
            (center_x + shoulder_width//2 + 35, shoulder_y + 80),
            (center_x + shoulder_width//2 + 30, waist_y),
            (center_x + shoulder_width//2 + 10, waist_y),
            (center_x + shoulder_width//2 + 15, shoulder_y + 80),
            (center_x + shoulder_width//2 + arm_width, shoulder_y)
        ]
        draw.polygon(right_arm_points, fill=skin_color, outline=(200, 180, 140), width=1)
        
        # Beine
        leg_width = 30
        # Linkes Bein
        left_leg_points = [
            (center_x - hip_width//2, hip_y),
            (center_x - leg_width//2, hip_y),
            (center_x - leg_width//2, knee_y),
            (center_x - leg_width//2 - 5, ankle_y),
            (center_x - leg_width//2 - 20, ankle_y),
            (center_x - leg_width//2 - 15, knee_y),
            (center_x - hip_width//2, knee_y)
        ]
        draw.polygon(left_leg_points, fill=skin_color, outline=(200, 180, 140), width=1)
        
        # Rechtes Bein
        right_leg_points = [
            (center_x + hip_width//2, hip_y),
            (center_x + leg_width//2, hip_y),
            (center_x + leg_width//2, knee_y),
            (center_x + leg_width//2 + 5, ankle_y),
            (center_x + leg_width//2 + 20, ankle_y),
            (center_x + leg_width//2 + 15, knee_y),
            (center_x + hip_width//2, knee_y)
        ]
        draw.polygon(right_leg_points, fill=skin_color, outline=(200, 180, 140), width=1)
        
        return img
    
    def apply_fashion_styling(self, model_img: Image.Image, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Wendet Fashion-Styling an"""
        draw = ImageDraw.Draw(model_img)
        width, height = model_img.size
        center_x = width // 2
        
        # Style-basierte Farben
        colors = self.get_style_colors(style_prompt)
        
        categories = [item['category'] for item in selected_items]
        
        for i, category in enumerate(set(categories)):
            color = colors[i % len(colors)]
            
            if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
                self.draw_detailed_top(draw, center_x, color)
            elif category == "Kleid":
                self.draw_detailed_dress(draw, center_x, color)
            elif category == "Hose":
                self.draw_detailed_pants(draw, center_x, color)
            elif category == "Mantel":
                self.draw_detailed_coat(draw, center_x, color)
            elif category in ["Sneaker", "Sandalen", "Stiefeletten"]:
                self.draw_detailed_shoes(draw, center_x, color, category)
        
        return model_img
    
    def draw_detailed_top(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet detailliertes Oberteil"""
        shoulder_y = 110
        waist_y = 270
        
        # Hauptteil mit Schatten
        shadow_color = tuple(max(0, c - 30) for c in color)
        
        # Schatten
        shadow_points = [
            (center_x - 32 + 3, shoulder_y + 3),
            (center_x + 32 + 3, shoulder_y + 3),
            (center_x + 27 + 3, waist_y + 3),
            (center_x - 27 + 3, waist_y + 3)
        ]
        draw.polygon(shadow_points, fill=shadow_color)
        
        # Hauptteil
        main_points = [
            (center_x - 32, shoulder_y),
            (center_x + 32, shoulder_y),
            (center_x + 27, waist_y),
            (center_x - 27, waist_y)
        ]
        draw.polygon(main_points, fill=color, outline=self.darken_color(color), width=2)
        
        # Ärmel mit Details
        # Linker Ärmel
        left_sleeve = [
            (center_x - 32, shoulder_y),
            (center_x - 50, shoulder_y + 25),
            (center_x - 45, shoulder_y + 60),
            (center_x - 32, shoulder_y + 45)
        ]
        draw.polygon(left_sleeve, fill=color, outline=self.darken_color(color), width=1)
        
        # Rechter Ärmel
        right_sleeve = [
            (center_x + 32, shoulder_y),
            (center_x + 50, shoulder_y + 25),
            (center_x + 45, shoulder_y + 60),
            (center_x + 32, shoulder_y + 45)
        ]
        draw.polygon(right_sleeve, fill=color, outline=self.darken_color(color), width=1)
        
        # Details (Nähte)
        draw.line([(center_x - 30, shoulder_y + 10), (center_x - 25, waist_y - 10)], 
                 fill=self.darken_color(color), width=1)
        draw.line([(center_x + 30, shoulder_y + 10), (center_x + 25, waist_y - 10)], 
                 fill=self.darken_color(color), width=1)
    
    def draw_detailed_dress(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet detailliertes Kleid"""
        shoulder_y = 110
        waist_y = 270
        hem_y = 500
        
        # Schatten
        shadow_color = tuple(max(0, c - 30) for c in color)
        shadow_points = [
            (center_x - 32 + 3, shoulder_y + 3),
            (center_x + 32 + 3, shoulder_y + 3),
            (center_x + 27 + 3, waist_y + 3),
            (center_x + 45 + 3, hem_y + 3),
            (center_x - 45 + 3, hem_y + 3),
            (center_x - 27 + 3, waist_y + 3)
        ]
        draw.polygon(shadow_points, fill=shadow_color)
        
        # Hauptkleid
        dress_points = [
            (center_x - 32, shoulder_y),
            (center_x + 32, shoulder_y),
            (center_x + 27, waist_y),
            (center_x + 45, hem_y),
            (center_x - 45, hem_y),
            (center_x - 27, waist_y)
        ]
        draw.polygon(dress_points, fill=color, outline=self.darken_color(color), width=2)
        
        # Ärmel
        self.draw_dress_sleeves(draw, center_x, shoulder_y, color)
        
        # Details
        # Taillierung
        draw.arc([center_x - 30, waist_y - 10, center_x + 30, waist_y + 10], 0, 180, 
                fill=self.darken_color(color), width=2)
    
    def draw_dress_sleeves(self, draw: ImageDraw.Draw, center_x: int, shoulder_y: int, color: Tuple[int, int, int]):
        """Zeichnet Kleid-Ärmel"""
        draw.polygon([
            (center_x - 32, shoulder_y),
            (center_x - 45, shoulder_y + 15),
            (center_x - 40, shoulder_y + 35),
            (center_x - 32, shoulder_y + 25)
        ], fill=color, outline=self.darken_color(color), width=1)
        
        draw.polygon([
            (center_x + 32, shoulder_y),
            (center_x + 45, shoulder_y + 15),
            (center_x + 40, shoulder_y + 35),
            (center_x + 32, shoulder_y + 25)
        ], fill=color, outline=self.darken_color(color), width=1)
    
    def draw_detailed_pants(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet detaillierte Hose"""
        waist_y = 270
        crotch_y = 360
        knee_y = 480
        ankle_y = 600
        
        # Schatten
        shadow_color = tuple(max(0, c - 30) for c in color)
        
        # Linkes Bein mit Schatten
        left_shadow = [
            (center_x - 27 + 2, waist_y + 2),
            (center_x - 2 + 2, waist_y + 2),
            (center_x - 2 + 2, crotch_y + 2),
            (center_x - 12 + 2, knee_y + 2),
            (center_x - 17 + 2, ankle_y + 2),
            (center_x - 32 + 2, ankle_y + 2),
            (center_x - 27 + 2, knee_y + 2),
            (center_x - 27 + 2, crotch_y + 2)
        ]
        draw.polygon(left_shadow, fill=shadow_color)
        
        # Rechtes Bein mit Schatten
        right_shadow = [
            (center_x + 2 + 2, waist_y + 2),
            (center_x + 27 + 2, waist_y + 2),
            (center_x + 27 + 2, crotch_y + 2),
            (center_x + 27 + 2, knee_y + 2),
            (center_x + 32 + 2, ankle_y + 2),
            (center_x + 17 + 2, ankle_y + 2),
            (center_x + 12 + 2, knee_y + 2),
            (center_x + 2 + 2, crotch_y + 2)
        ]
        draw.polygon(right_shadow, fill=shadow_color)
        
        # Haupthose
        # Linkes Bein
        left_leg = [
            (center_x - 27, waist_y),
            (center_x - 2, waist_y),
            (center_x - 2, crotch_y),
            (center_x - 12, knee_y),
            (center_x - 17, ankle_y),
            (center_x - 32, ankle_y),
            (center_x - 27, knee_y),
            (center_x - 27, crotch_y)
        ]
        draw.polygon(left_leg, fill=color, outline=self.darken_color(color), width=2)
        
        # Rechtes Bein
        right_leg = [
            (center_x + 2, waist_y),
            (center_x + 27, waist_y),
            (center_x + 27, crotch_y),
            (center_x + 27, knee_y),
            (center_x + 32, ankle_y),
            (center_x + 17, ankle_y),
            (center_x + 12, knee_y),
            (center_x + 2, crotch_y)
        ]
        draw.polygon(right_leg, fill=color, outline=self.darken_color(color), width=2)
        
        # Details
        # Mittelnaht
        draw.line([(center_x, waist_y), (center_x, crotch_y)], 
                 fill=self.darken_color(color), width=2)
    
    def draw_detailed_coat(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet detaillierten Mantel"""
        shoulder_y = 100
        waist_y = 270
        hem_y = 480
        
        # Schatten
        shadow_color = tuple(max(0, c - 30) for c in color)
        shadow_points = [
            (center_x - 40 + 3, shoulder_y + 3),
            (center_x + 40 + 3, shoulder_y + 3),
            (center_x + 35 + 3, waist_y + 3),
            (center_x + 55 + 3, hem_y + 3),
            (center_x - 55 + 3, hem_y + 3),
            (center_x - 35 + 3, waist_y + 3)
        ]
        draw.polygon(shadow_points, fill=shadow_color)
        
        # Hauptmantel
        coat_points = [
            (center_x - 40, shoulder_y),
            (center_x + 40, shoulder_y),
            (center_x + 35, waist_y),
            (center_x + 55, hem_y),
            (center_x - 55, hem_y),
            (center_x - 35, waist_y)
        ]
        draw.polygon(coat_points, fill=color, outline=self.darken_color(color), width=3)
        
        # Mantel-Ärmel
        # Linker Ärmel
        left_sleeve = [
            (center_x - 40, shoulder_y),
            (center_x - 65, shoulder_y + 30),
            (center_x - 60, shoulder_y + 100),
            (center_x - 40, shoulder_y + 80)
        ]
        draw.polygon(left_sleeve, fill=color, outline=self.darken_color(color), width=2)
        
        # Rechter Ärmel
        right_sleeve = [
            (center_x + 40, shoulder_y),
            (center_x + 65, shoulder_y + 30),
            (center_x + 60, shoulder_y + 100),
            (center_x + 40, shoulder_y + 80)
        ]
        draw.polygon(right_sleeve, fill=color, outline=self.darken_color(color), width=2)
        
        # Knöpfe
        for i in range(4):
            button_y = shoulder_y + 50 + i * 35
            draw.ellipse([center_x - 4, button_y, center_x + 4, button_y + 8], 
                        fill=self.darken_color(color))
        
        # Kragen
        collar_points = [
            (center_x - 15, shoulder_y - 5),
            (center_x + 15, shoulder_y - 5),
            (center_x + 20, shoulder_y + 15),
            (center_x - 20, shoulder_y + 15)
        ]
        draw.polygon(collar_points, fill=self.darken_color(color))
    
    def draw_detailed_shoes(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], shoe_type: str):
        """Zeichnet detaillierte Schuhe"""
        ankle_y = 600
        
        if shoe_type == "Stiefeletten":
            # Stiefeletten mit Details
            boot_height = 60
            
            # Schatten
            shadow_color = tuple(max(0, c - 40) for c in color)
            draw.polygon([
                (center_x - 30 + 2, ankle_y - boot_height + 2),
                (center_x - 12 + 2, ankle_y - boot_height + 2),
                (center_x - 8 + 2, ankle_y + 2),
                (center_x - 35 + 2, ankle_y + 2),
                (center_x - 38 + 2, ankle_y - 15 + 2)
            ], fill=shadow_color)
            
            draw.polygon([
                (center_x + 12 + 2, ankle_y - boot_height + 2),
                (center_x + 30 + 2, ankle_y - boot_height + 2),
                (center_x + 38 + 2, ankle_y - 15 + 2),
                (center_x + 35 + 2, ankle_y + 2),
                (center_x + 8 + 2, ankle_y + 2)
            ], fill=shadow_color)
            
            # Linker Stiefel
            draw.polygon([
                (center_x - 30, ankle_y - boot_height),
                (center_x - 12, ankle_y - boot_height),
                (center_x - 8, ankle_y),
                (center_x - 35, ankle_y),
                (center_x - 38, ankle_y - 15)
            ], fill=color, outline=self.darken_color(color), width=2)
            
            # Rechter Stiefel
            draw.polygon([
                (center_x + 12, ankle_y - boot_height),
                (center_x + 30, ankle_y - boot_height),
                (center_x + 38, ankle_y - 15),
                (center_x + 35, ankle_y),
                (center_x + 8, ankle_y)
            ], fill=color, outline=self.darken_color(color), width=2)
            
            # Details (Schnürung)
            for i in range(3):
                detail_y = ankle_y - boot_height + 15 + i * 15
                draw.line([(center_x - 25, detail_y), (center_x - 15, detail_y)], 
                         fill=self.darken_color(color), width=1)
                draw.line([(center_x + 15, detail_y), (center_x + 25, detail_y)], 
                         fill=self.darken_color(color), width=1)
        
        else:
            # Normale Schuhe mit Details
            # Schatten
            shadow_color = tuple(max(0, c - 40) for c in color)
            draw.ellipse([center_x - 30 + 2, ankle_y - 12 + 2, center_x - 2 + 2, ankle_y + 12 + 2], 
                        fill=shadow_color)
            draw.ellipse([center_x + 2 + 2, ankle_y - 12 + 2, center_x + 30 + 2, ankle_y + 12 + 2], 
                        fill=shadow_color)
            
            # Linker Schuh
            draw.ellipse([center_x - 30, ankle_y - 12, center_x - 2, ankle_y + 12], 
                        fill=color, outline=self.darken_color(color), width=2)
            # Rechter Schuh
            draw.ellipse([center_x + 2, ankle_y - 12, center_x + 30, ankle_y + 12], 
                        fill=color, outline=self.darken_color(color), width=2)
            
            # Schuhdetails
            draw.arc([center_x - 28, ankle_y - 8, center_x - 4, ankle_y + 8], 0, 180, 
                    fill=self.darken_color(color), width=1)
            draw.arc([center_x + 4, ankle_y - 8, center_x + 28, ankle_y + 8], 0, 180, 
                    fill=self.darken_color(color), width=1)
    
    def add_professional_effects(self, img: Image.Image) -> Image.Image:
        """Fügt professionelle Effekte hinzu"""
        # Weiche Schatten
        shadow_img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Beleuchtungs-Overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = img.size
        center_x, center_y = width // 2, height // 3
        
        # Hauptlicht
        for radius in range(150, 0, -8):
            alpha = int(12 * (150 - radius) / 150)
            draw.ellipse([
                center_x - 80 - radius//2, center_y - 80 - radius//2,
                center_x - 80 + radius//2, center_y - 80 + radius//2
            ], fill=(255, 255, 255, alpha))
        
        # Composite
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        # Schärfe verbessern
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        return img
    
    def get_style_colors(self, style_prompt: str) -> List[Tuple[int, int, int]]:
        """Gibt Farben basierend auf Style zurück"""
        style_lower = style_prompt.lower()
        
        if "modern" in style_lower or "minimalistisch" in style_lower:
            return [(45, 45, 48), (240, 240, 245), (128, 128, 130)]
        elif "klassisch" in style_lower:
            return [(25, 25, 112), (255, 255, 255), (139, 69, 19)]
        elif "vintage" in style_lower:
            return [(139, 69, 19), (205, 133, 63), (128, 0, 0)]
        elif "sportlich" in style_lower:
            return [(0, 0, 0), (255, 69, 0), (255, 255, 255)]
        elif "extravagant" in style_lower:
            return [(148, 0, 211), (255, 20, 147), (255, 215, 0)]
        elif "natürlich" in style_lower:
            return [(34, 139, 34), (139, 90, 43), (240, 230, 140)]
        elif "monochrom" in style_lower:
            return [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
        elif "pastell" in style_lower:
            return [(255, 182, 193), (176, 224, 230), (221, 160, 221)]
        else:
            return [(47, 79, 79), (205, 192, 176), (139, 69, 19)]
    
    def darken_color(self, color: Tuple[int, int, int], factor: float = 0.6) -> Tuple[int, int, int]:
        """Dunkelt eine Farbe ab"""
        return tuple(max(0, int(c * factor)) for c in color[:3])

def create_professional_fashion_prompt(selected_items: List[Dict], style_prompt: str) -> str:
    """Erstellt professionelle Fashion-Photography Prompts"""
    categories = [item['category'] for item in selected_items]
    category_text = ", ".join(set(categories))
    
    base_prompt = f"professional fashion photography, elegant female model wearing {category_text}"
    
    style_additions = {
        "modern": "contemporary minimalist design, clean lines, urban setting",
        "klassisch": "timeless elegant style, refined sophisticated look",
        "vintage": "retro inspired fashion, classic vintage aesthetic",
        "sportlich": "athletic wear, dynamic pose, fitness lifestyle",
        "extravagant": "high fashion couture, dramatic styling, avant-garde",
        "minimalistisch": "clean minimal aesthetic, neutral colors, simple elegance",
        "natürlich": "organic natural materials, earth tones, relaxed fit",
        "monochrom": "black and white styling, monochromatic palette",
        "pastell": "soft pastel colors, dreamy romantic style"
    }
    
    style_addition = ""
    for key, addition in style_additions.items():
        if key in style_prompt.lower():
            style_addition = addition
            break
    
    full_prompt = f"{base_prompt}, {style_addition}, studio lighting, high fashion photography, professional model pose, detailed fabric textures, photorealistic, 8k quality, magazine cover quality"
    
    return full_prompt

def generate_fashion_design_optimized(selected_items: List[Dict], style_prompt: str, 
                                    generator: CloudOptimizedFashionGenerator) -> Optional[np.ndarray]:
    """Optimierte Generierung für Streamlit Cloud"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🎨 Erstelle Fashion-Prompt...")
        progress_bar.progress(0.1)
        
        prompt = create_professional_fashion_prompt(selected_items, style_prompt)
        
        st.write(f"**Generierungs-Prompt:** {prompt[:100]}...")
        
        # API Verfügbarkeit prüfen
        available_apis = generator.check_api_availability()
        
        generated_image = None
        
        # Versuche APIs in Reihenfolge der Zuverlässigkeit
        if available_apis['stability']:
            status_text.text("🎯 Generiere mit Stability AI (Premium)...")
            progress_bar.progress(0.3)
            generated_image = generator.generate_with_stability_api(prompt)
        
        if generated_image is None and available_apis['replicate']:
            status_text.text("🔄 Versuche Replicate API...")
            progress_bar.progress(0.5)
            generated_image = generator.generate_with_replicate_api(prompt)
        
        if generated_image is None and available_apis['huggingface']:
            status_text.text("🤗 Versuche Hugging Face API...")
            progress_bar.progress(0.7)
            generated_image = generator.generate_with_huggingface_inference(prompt)
        
        if generated_image is None:
            # Fallback zu hochwertiger lokaler Generierung
            status_text.text("🎨 Erstelle hochwertiges Fashion-Design...")
            progress_bar.progress(0.9)
            generated_image = generator.create_professional_fashion_design(selected_items, style_prompt)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Fashion-Design fertig!")
        
        return np.array(generated_image).astype(np.float32) / 255.0
        
    except Exception as e:
        st.error(f"Fehler bei der Generierung: {e}")
        # Fallback
        status_text.text("🎨 Erstelle Fallback-Design...")
        try:
            fallback_image = generator.create_professional_fashion_design(selected_items, style_prompt)
            return np.array(fallback_image).astype(np.float32) / 255.0
        except Exception:
            return None

# Utility Functions
def image_to_base64(image_array):
    """Konvertiert Bild zu Base64"""
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
    image.save(buffer, format="PNG", optimize=True, quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@st.cache_data(max_entries=1, ttl=3600)
def load_fashion_mnist_optimized(sample_size=500):
    """Lädt Fashion-MNIST mit Caching"""
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        indices = np.random.choice(len(x_train), size=min(sample_size, len(x_train)), replace=False)
        x_sample = x_train[indices]
        y_sample = y_train[indices]
        
        del x_train, y_train, x_test, y_test
        gc.collect()
        
        return x_sample, y_sample
    except Exception as e:
        st.error(f"Fehler beim Laden von Fashion-MNIST: {e}")
        return None, None

@st.cache_data(max_entries=1, ttl=1800)
def select_random_fashion_items():
    """Wählt zufällige Fashion-Items mit Caching"""
    x_all, y_all = load_fashion_mnist_optimized(sample_size=300)
    
    if x_all is None:
        return []
    
    num_items = min(15, len(x_all))  # Weniger Items für bessere Performance
    random_indices = random.sample(range(len(x_all)), num_items)
    
    items = []
    brands = ["Chanel", "Dior", "Versace", "Prada", "Gucci", "Armani", "Zara", "H&M", "COS", "Uniqlo"]
    prices = ["45€", "65€", "85€", "120€", "150€", "200€", "250€", "300€"]
    
    for i, idx in enumerate(random_indices):
        image = x_all[idx]
        label = y_all[idx]
        category = FASHION_CLASSES[label]
        
        item = {
            "id": i + 1,
            "name": f"{category} Collection #{idx}",
            "brand": random.choice(brands),
            "price": random.choice(prices),
            "category": category,
            "description": f"Premium {category.lower()} aus der aktuellen Kollektion",
            "image_data": image_to_base64(image),
            "original_index": idx,
            "label": label,
            "timestamp": datetime.now().isoformat()
        }
        items.append(item)
    
    del x_all, y_all
    gc.collect()
    
    return items

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
        with st.spinner("Lade Fashion-Kollektion..."):
            st.session_state.fashion_items = select_random_fashion_items()
    if 'all_time_favorites' not in st.session_state:
        st.session_state.all_time_favorites = []
    if 'selected_for_generation' not in st.session_state:
        st.session_state.selected_for_generation = []
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'generator' not in st.session_state:
        st.session_state.generator = CloudOptimizedFashionGenerator()
    
    # Limitiere gespeicherte Daten für bessere Performance
    if len(st.session_state.generated_images) > 10:
        st.session_state.generated_images = st.session_state.generated_images[-10:]
    if len(st.session_state.all_time_favorites) > 30:
        st.session_state.all_time_favorites = st.session_state.all_time_favorites[-30:]

# UI Functions
def like_item():
    """Liked ein Item"""
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        current_item = items[st.session_state.current_index]
        st.session_state.liked_items.append(current_item)
        
        if len(st.session_state.all_time_favorites) < 30:
            if not any(fav['original_index'] == current_item['original_index'] 
                      for fav in st.session_state.all_time_favorites):
                st.session_state.all_time_favorites.append(current_item)
        
        st.session_state.current_index += 1

def dislike_item():
    """Disliked ein Item"""
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        st.session_state.current_index += 1

def reset_session():
    """Setzt Session zurück"""
    st.session_state.current_index = 0
    st.session_state.liked_items = []
    st.session_state.disliked_items = []
    gc.collect()
    
    with st.spinner("Lade neue Fashion-Kollektion..."):
        # Cache leeren
        select_random_fashion_items.clear()
        st.session_state.fashion_items = select_random_fashion_items()

# UI Rendering Functions
def render_swipe_tab():
    """Rendert den Swipe-Tab"""
    items = st.session_state.fashion_items
    current_idx = st.session_state.current_index
    total_items = len(items)
    
    if not items:
        st.error("Fehler beim Laden der Fashion-Daten.")
        if st.button("🔄 Erneut versuchen"):
            st.rerun()
        return
    
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
            if st.button("👎 Skip", type="secondary", use_container_width=True, help="Nicht interessant"):
                dislike_item()
                st.rerun()
        
        with col3:
            if st.button("❤️ Love it!", type="primary", use_container_width=True, help="Zu Favoriten hinzufügen"):
                like_item()
                st.rerun()

def render_generate_tab():
    """Rendert den Generate-Tab"""
    st.markdown("## 🎨 Fashion-Generator")
    
    # Info über API-Verfügbarkeit
    generator = st.session_state.generator
    available_apis = generator.check_api_availability()
    
    if any(available_apis.values()):
        api_info = []
        if available_apis['stability']:
            api_info.append("✅ Stability AI (Premium)")
        if available_apis['replicate']:
            api_info.append("✅ Replicate")
        if available_apis['huggingface']:
            api_info.append("✅ Hugging Face")
        
        st.markdown(f"""
        <div class="generation-alert">
            <h3>🚀 AI-APIs verfügbar</h3>
            <p>Folgende Premium-APIs sind konfiguriert:<br>
            {', '.join(api_info)}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="generation-alert">
            <h3>💡 Lokaler Modus</h3>
            <p>Keine externen APIs konfiguriert. Die App erstellt hochwertige Fashion-Designs 
            mit optimierten lokalen Algorithmen.</p>
            <details>
                <summary>🔑 API-Konfiguration (optional)</summary>
                <p>Für AI-generierte Bilder, füge folgende Secrets in Streamlit Cloud hinzu:</p>
                <ul>
                    <li><code>STABILITY_API_KEY</code> - Stability AI</li>
                    <li><code>REPLICATE_API_TOKEN</code> - Replicate</li>
                    <li><code>HUGGINGFACE_TOKEN</code> - Hugging Face</li>
                </ul>
            </details>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.all_time_favorites:
        st.warning("💡 Sammle erst Favoriten durch Swipen in der Fashion-Kollektion!")
        return
    
    # Favoriten-Auswahl
    st.markdown("### 👗 Wähle Fashion-Styles (max. 3 für beste Ergebnisse)")
    
    cols_per_row = 5
    items_to_show = st.session_state.all_time_favorites[:15]
    
    for row in range(0, len(items_to_show), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, item in enumerate(items_to_show[row:row + cols_per_row]):
            with cols[idx]:
                is_selected = item in st.session_state.selected_for_generation
                
                button_style = "✅" if is_selected else "⭕"
                button_color = "primary" if is_selected else "secondary"
                
                if st.button(
                    f"{button_style}",
                    key=f"sel_{row}_{idx}",
                    type=button_color,
                    help=f"{item['category']} - {item['brand']}",
                    use_container_width=True
                ):
                    if is_selected:
                        st.session_state.selected_for_generation.remove(item)
                    else:
                        if len(st.session_state.selected_for_generation) < 3:
                            st.session_state.selected_for_generation.append(item)
                        else:
                            st.warning("⚠️ Maximal 3 Styles für optimale Ergebnisse!")
                    st.rerun()
                
                st.markdown(f"""
                <div style="text-align: center; margin-top: 5px;">
                    <img src="{item['image_data']}" 
                         style="width: 100%; border-radius: 8px; margin-bottom: 5px;
                                border: {'3px solid #4CAF50' if is_selected else '1px solid #ddd'};">
                    <div style="font-size: 0.8rem; color: #666;">{item['category']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    if st.session_state.selected_for_generation:
        st.markdown(f"""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <strong>✨ Ausgewählt:</strong> {len(st.session_state.selected_for_generation)}/3 Styles<br>
            <em>Styles: {', '.join([item['category'] for item in st.session_state.selected_for_generation])}</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Style-Konfiguration
        st.markdown("### 🎨 Fashion-Style Konfiguration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            style_mood = st.selectbox(
                "🎭 Style-Richtung",
                ["Modern", "Klassisch", "Vintage", "Sportlich", "Minimalistisch", "Extravagant"],
                help="Bestimmt den grundlegenden Look"
            )
        
        with col2:
            color_scheme = st.selectbox(
                "🎨 Farbschema",
                ["Natürlich", "Monochrom", "Pastell", "Kräftig", "Erdtöne"],
                help="Farbpalette für die Kleidung"
            )
        
        style_prompt = f"{style_mood.lower()} style, {color_scheme.lower()} colors"
        
        # Generate-Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎨 ✨ Generiere Fashion-Design!", type="primary", use_container_width=True):
                with st.spinner("Erstelle professionelles Fashion-Design..."):
                    generated_image = generate_fashion_design_optimized(
                        st.session_state.selected_for_generation,
                        style_prompt,
                        st.session_state.generator
                    )
                    
                    if generated_image is not None:
                        # Speichere Generierung
                        generation_data = {
                            'image': numpy_to_base64(generated_image, size=(512, 768)),
                            'style': f"{style_mood}",
                            'colors': color_scheme,
                            'items': [item['category'] for item in st.session_state.selected_for_generation],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.session_state.generated_images.append(generation_data)
                        if len(st.session_state.generated_images) > 10:
                            st.session_state.generated_images.pop(0)
                        
                        # Darstellung des Ergebnisses
                        st.markdown("---")
                        st.markdown("## 📸 Generiertes Fashion-Design")
                        
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("""
                            <div class="runway-preview">
                                <div style="position: relative; z-index: 1;">
                            """, unsafe_allow_html=True)
                            
                            st.image(generated_image, 
                                   caption="Fashion-Design (AI-generiert)",
                                   use_column_width=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: white; padding: 20px; border-radius: 15px; 
                                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                                <h4>📋 Design-Details</h4>
                                <p><strong>Style:</strong> {style_mood}</p>
                                <p><strong>Farben:</strong> {color_scheme}</p>
                                <p><strong>Fashion-Items:</strong><br>
                                   {', '.join([item['category'] for item in st.session_state.selected_for_generation])}</p>
                                <p><strong>Generiert:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Download-Button
                            img_buffer = io.BytesIO()
                            img = Image.fromarray((generated_image * 255).astype(np.uint8))
                            img.save(img_buffer, format="PNG", optimize=True, quality=85)
                            img_buffer.seek(0)
                            
                            st.download_button(
                                "💾 Design herunterladen (PNG)",
                                data=img_buffer,
                                file_name=f"fashion_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ Fehler bei der Generierung. Bitte versuche es erneut.")

def render_gallery_tab():
    """Zeigt Galerie der generierten Designs"""
    st.markdown("## 🖼️ Fashion-Design Galerie")
    
    if not st.session_state.generated_images:
        st.markdown("""
        <div style="text-align: center; padding: 60px; color: #666;">
            <h3>📷 Noch keine Fashion-Designs generiert</h3>
            <p>Gehe zum Generator-Tab und erstelle dein erstes Fashion-Design!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"**📸 {len(st.session_state.generated_images)} generierte Fashion-Designs**")
    
    # Galerie-Darstellung
    for idx, gen_data in enumerate(reversed(st.session_state.generated_images)):
        with st.container():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.image(gen_data['image'], use_column_width=True)
            
            with col2:
                st.markdown(f"""
                <div style="padding: 20px;">
                    <h4>📸 Fashion-Design #{len(st.session_state.generated_images) - idx}</h4>
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
    <div class="sub-header">Cloud-optimierter Fashion Generator</div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔄 Fashion Swipe", "🎨 AI Generator", "🖼️ Design-Galerie"])
    
    with tab1:
        render_swipe_tab()
    
    with tab2:
        render_generate_tab()
    
    with tab3:
        render_gallery_tab()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Session-Statistiken")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("❤️ Likes", len(st.session_state.liked_items))
        with col2:
            st.metric("⭐ Favoriten", len(st.session_state.all_time_favorites))
        
        st.metric("📸 Generierte Designs", len(st.session_state.generated_images))
        
        st.markdown("---")
        
        # Action Buttons
        if st.button("🔄 Neue Session", use_container_width=True):
            reset_session()
            st.rerun()
        
        if st.button("🗑️ Galerie leeren", use_container_width=True):
            st.session_state.generated_images = []
            st.rerun()
        
        st.markdown("---")
        
        # Cloud Info
        st.markdown("""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; font-size: 0.8rem;">
            <strong>☁️ Streamlit Cloud Optimiert:</strong><br>
            • Speicher-optimiert (< 1GB RAM)<br>
            • API-Integration verfügbar<br>
            • Lokaler Fallback-Modus<br>
            • Cached Fashion-MNIST
        </div>
        """, unsafe_allow_html=True)
        
        # API Status
        if st.checkbox("🔧 API Status"):
            generator = st.session_state.generator
            available_apis = generator.check_api_availability()
            
            for api_name, is_available in available_apis.items():
                status = "✅ Verfügbar" if is_available else "❌ Nicht konfiguriert"
                st.caption(f"{api_name.title()}: {status}")
        
        # Memory Info
        if st.checkbox("💾 Memory Info"):
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                st.caption(f"RAM: {memory_mb:.1f} MB")
            except ImportError:
                st.caption("psutil nicht verfügbar")

if __name__ == "__main__":
    # Memory Management für Streamlit Cloud
    try:
        import resource
        # Limitiere Memory auf 1GB
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))
    except (ImportError, AttributeError, OSError):
        pass
    
    # Garbage Collection konfigurieren
    gc.set_threshold(700, 10, 10)
    
    main()
