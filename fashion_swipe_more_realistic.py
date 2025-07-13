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

class PremiumFashionGenerator:
    """Premium Fashion Generator mit AI-APIs"""
    
    def __init__(self):
        self.available_apis = self.check_api_availability()
        
    def check_api_availability(self) -> Dict[str, bool]:
        """Prüft verfügbare APIs"""
        apis = {
            'stability': bool(os.getenv('STABILITY_API_KEY')),
            'replicate': bool(os.getenv('REPLICATE_API_TOKEN')),
            'huggingface': bool(os.getenv('HUGGINGFACE_TOKEN')),
            'openai': bool(os.getenv('OPENAI_API_KEY'))
        }
        return apis
    
    def generate_with_stability_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt Stability AI API - Beste Qualität"""
        api_key = os.getenv('STABILITY_API_KEY')
        if not api_key:
            return None
            
        try:
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
                        "text": "blurry, bad quality, distorted, amateur, low resolution, cartoon, anime",
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
                
        except Exception as e:
            st.warning(f"Stability API Fehler: {e}")
        
        return None
    
    def generate_with_replicate_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt Replicate API - Gute Alternative"""
        api_token = os.getenv('REPLICATE_API_TOKEN')
        if not api_token:
            return None
            
        try:
            headers = {
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json"
            }
            
            # SDXL über Replicate
            data = {
                "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {
                    "prompt": prompt,
                    "negative_prompt": "blurry, bad quality, distorted, amateur, cartoon, anime",
                    "width": 512,
                    "height": 768,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 25,
                    "scheduler": "K_EULER"
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
                
                # Warte auf Ergebnis
                for i in range(60):  # Max 60 Sekunden
                    time.sleep(1)
                    result_response = requests.get(prediction_url, headers=headers, timeout=5)
                    result = result_response.json()
                    
                    if result["status"] == "succeeded" and result.get("output"):
                        image_url = result["output"][0]
                        img_response = requests.get(image_url, timeout=10)
                        if img_response.status_code == 200:
                            return Image.open(io.BytesIO(img_response.content))
                    elif result["status"] == "failed":
                        break
                    
                    # Progress anzeigen
                    if i % 10 == 0:
                        st.info(f"⏳ Generierung läuft... ({i}/60s)")
            
        except Exception as e:
            st.warning(f"Replicate API Fehler: {e}")
        
        return None
    
    def generate_with_huggingface_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt Hugging Face API - Kostenlos aber langsamer"""
        api_token = os.getenv('HUGGINGFACE_TOKEN')
        if not api_token:
            return None
        
        try:
            # Beste verfügbare Modelle
            models = [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "prompthero/openjourney-v4"
            ]
            
            for model in models:
                try:
                    API_URL = f"https://api-inference.huggingface.co/models/{model}"
                    headers = {"Authorization": f"Bearer {api_token}"}
                    
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "negative_prompt": "blurry, bad quality, distorted, amateur, cartoon",
                            "num_inference_steps": 20,
                            "guidance_scale": 7.5,
                            "width": 512,
                            "height": 768
                        }
                    }
                    
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                        return Image.open(io.BytesIO(response.content))
                    elif response.status_code == 503:
                        st.info(f"⏳ Model {model.split('/')[-1]} lädt... versuche nächstes Modell")
                        continue
                        
                except Exception:
                    continue
            
        except Exception as e:
            st.warning(f"Hugging Face API Fehler: {e}")
        
        return None
    
    def generate_with_openai_dalle(self, prompt: str) -> Optional[Image.Image]:
        """Nutzt OpenAI DALL-E - Premium Option"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1792",
                "quality": "hd",
                "style": "natural"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                image_url = result["data"][0]["url"]
                img_response = requests.get(image_url, timeout=10)
                if img_response.status_code == 200:
                    return Image.open(io.BytesIO(img_response.content))
                    
        except Exception as e:
            st.warning(f"OpenAI DALL-E Fehler: {e}")
        
        return None
    
    def create_enhanced_fallback_design(self, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Verbesserter Fallback mit besserer Qualität"""
        
        # Hochauflösender Canvas
        img = Image.new('RGB', (512, 768), color=(245, 245, 250))
        
        # Professioneller Hintergrund mit Texturen
        img = self.create_textured_background(img, style_prompt)
        
        # Detailliertes Model
        model_img = self.create_detailed_model(img.size)
        
        # Styling mit besseren Details
        model_img = self.apply_premium_styling(model_img, selected_items, style_prompt)
        
        # Professionelle Effekte
        model_img = self.add_professional_effects(model_img)
        
        # Combine mit Blending
        final_img = Image.alpha_composite(img.convert('RGBA'), model_img.convert('RGBA'))
        
        # Post-Processing
        final_img = self.enhance_image_quality(final_img.convert('RGB'))
        
        return final_img
    
    def create_textured_background(self, base_img: Image.Image, style_prompt: str) -> Image.Image:
        """Erstellt texturierten Hintergrund basierend auf Style"""
        draw = ImageDraw.Draw(base_img)
        width, height = base_img.size
        
        # Style-basierte Hintergründe
        if "vintage" in style_prompt.lower():
            # Vintage-Textur
            for y in range(0, height, 5):
                for x in range(0, width, 5):
                    shade = random.randint(220, 240)
                    draw.rectangle([x, y, x+5, y+5], fill=(shade, shade-10, shade-5))
        elif "modern" in style_prompt.lower():
            # Moderne Geometrie
            for i in range(0, width, 40):
                alpha = int(255 * 0.1)
                overlay = Image.new('RGBA', base_img.size, (255, 255, 255, alpha))
                base_img = Image.alpha_composite(base_img.convert('RGBA'), overlay).convert('RGB')
        else:
            # Standard Studio-Gradient
            for y in range(height):
                shade = int(240 + (y / height) * 15)
                draw.rectangle([0, y, width, y+1], fill=(shade, shade, shade+5))
        
        # Beleuchtungseffekte
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
        light_draw = ImageDraw.Draw(overlay)
        
        # Mehrere Lichtquellen
        for light_x, light_y, intensity in [(width//3, height//4, 20), (2*width//3, height//3, 15)]:
            for radius in range(150, 0, -10):
                alpha = int(intensity * (150 - radius) / 150)
                light_draw.ellipse([
                    light_x - radius, light_y - radius//2,
                    light_x + radius, light_y + radius//2
                ], fill=(255, 255, 255, alpha))
        
        base_img = Image.alpha_composite(base_img.convert('RGBA'), overlay).convert('RGB')
        return base_img
    
    def create_detailed_model(self, size: Tuple[int, int]) -> Image.Image:
        """Erstellt detaillierteres Model"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center_x = width // 2
        skin_color = (245, 220, 177, 255)
        
        # Proportionen (realistischer)
        head_y = height // 8
        neck_y = head_y + 40
        shoulder_y = neck_y + 25
        chest_y = shoulder_y + 70
        waist_y = chest_y + 90
        hip_y = waist_y + 70
        knee_y = hip_y + 140
        ankle_y = height - 80
        
        # Kopf mit Details
        head_width, head_height = 45, 60
        draw.ellipse([
            center_x - head_width//2, head_y - head_height//2,
            center_x + head_width//2, head_y + head_height//2
        ], fill=skin_color, outline=(200, 180, 140, 255), width=2)
        
        # Gesichtszüge
        eye_y = head_y - 8
        # Augen
        draw.ellipse([center_x - 16, eye_y - 3, center_x - 10, eye_y + 3], fill=(70, 50, 30, 255))
        draw.ellipse([center_x + 10, eye_y - 3, center_x + 16, eye_y + 3], fill=(70, 50, 30, 255))
        # Nase
        draw.polygon([(center_x - 2, head_y), (center_x + 2, head_y), (center_x, head_y + 8)], fill=(220, 200, 160, 255))
        # Mund
        draw.arc([center_x - 6, head_y + 15, center_x + 6, head_y + 22], 0, 180, fill=(200, 120, 120, 255), width=2)
        
        # Haare (detaillierter)
        hair_color = (101, 67, 33, 255)
        # Haupthaar
        hair_points = [
            (center_x - head_width//2 - 5, head_y - head_height//2 - 10),
            (center_x + head_width//2 + 5, head_y - head_height//2 - 10),
            (center_x + head_width//2 + 3, head_y + head_height//2 - 15),
            (center_x - head_width//2 - 3, head_y + head_height//2 - 15)
        ]
        draw.polygon(hair_points, fill=hair_color)
        
        # Haarsträhnen
        for i in range(5):
            strand_x = center_x - 15 + i * 6
            draw.line([(strand_x, head_y - head_height//2 - 8), 
                      (strand_x + random.randint(-2, 2), head_y - head_height//2 + 5)], 
                     fill=hair_color, width=2)
        
        # Hals
        draw.rectangle([center_x - 10, neck_y, center_x + 10, shoulder_y], fill=skin_color)
        
        # Körper (realistischere Proportionen)
        shoulder_width = 55
        chest_width = 50
        waist_width = 40
        hip_width = 55
        
        # Torso mit natürlicher Kurve
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
        draw.polygon(torso_points, fill=skin_color, outline=(200, 180, 140, 255), width=1)
        
        # Arme (natürlicher)
        arm_width = 18
        # Linker Arm
        left_arm_points = [
            (center_x - shoulder_width//2, shoulder_y),
            (center_x - shoulder_width//2 - 30, shoulder_y + 60),
            (center_x - shoulder_width//2 - 25, waist_y - 20),
            (center_x - shoulder_width//2 - 5, waist_y - 20),
            (center_x - shoulder_width//2 - 10, shoulder_y + 60),
            (center_x - shoulder_width//2 - arm_width, shoulder_y)
        ]
        draw.polygon(left_arm_points, fill=skin_color, outline=(200, 180, 140, 255), width=1)
        
        # Rechter Arm
        right_arm_points = [
            (center_x + shoulder_width//2, shoulder_y),
            (center_x + shoulder_width//2 + 30, shoulder_y + 60),
            (center_x + shoulder_width//2 + 25, waist_y - 20),
            (center_x + shoulder_width//2 + 5, waist_y - 20),
            (center_x + shoulder_width//2 + 10, shoulder_y + 60),
            (center_x + shoulder_width//2 + arm_width, shoulder_y)
        ]
        draw.polygon(right_arm_points, fill=skin_color, outline=(200, 180, 140, 255), width=1)
        
        # Beine (realistischer)
        leg_width = 25
        # Linkes Bein
        left_leg_points = [
            (center_x - hip_width//2, hip_y),
            (center_x - leg_width//2, hip_y),
            (center_x - leg_width//2, knee_y),
            (center_x - leg_width//2 - 3, ankle_y),
            (center_x - leg_width//2 - 18, ankle_y),
            (center_x - leg_width//2 - 15, knee_y),
            (center_x - hip_width//2, knee_y)
        ]
        draw.polygon(left_leg_points, fill=skin_color, outline=(200, 180, 140, 255), width=1)
        
        # Rechtes Bein
        right_leg_points = [
            (center_x + hip_width//2, hip_y),
            (center_x + leg_width//2, hip_y),
            (center_x + leg_width//2, knee_y),
            (center_x + leg_width//2 + 3, ankle_y),
            (center_x + leg_width//2 + 18, ankle_y),
            (center_x + leg_width//2 + 15, knee_y),
            (center_x + hip_width//2, knee_y)
        ]
        draw.polygon(right_leg_points, fill=skin_color, outline=(200, 180, 140, 255), width=1)
        
        return img
    
    def apply_premium_styling(self, model_img: Image.Image, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Premium Styling mit Details"""
        draw = ImageDraw.Draw(model_img)
        width, height = model_img.size
        center_x = width // 2
        
        # Style-basierte Farben
        colors = self.get_premium_colors(style_prompt)
        
        categories = [item['category'] for item in selected_items]
        
        for i, category in enumerate(set(categories)):
            color = colors[i % len(colors)]
            
            if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
                self.draw_premium_top(draw, center_x, color, style_prompt)
            elif category == "Kleid":
                self.draw_premium_dress(draw, center_x, color, style_prompt)
            elif category == "Hose":
                self.draw_premium_pants(draw, center_x, color, style_prompt)
            elif category == "Mantel":
                self.draw_premium_coat(draw, center_x, color, style_prompt)
            elif category in ["Sneaker", "Sandalen", "Stiefeletten"]:
                self.draw_premium_shoes(draw, center_x, color, category, style_prompt)
        
        return model_img
    
    def draw_premium_top(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], style: str):
        """Premium Oberteil mit Details"""
        shoulder_y = 95
        waist_y = 235
        
        # Schatten
        shadow_color = tuple(max(0, c - 40) for c in color) + (120,)
        shadow_points = [
            (center_x - 25 + 2, shoulder_y + 2),
            (center_x + 25 + 2, shoulder_y + 2),
            (center_x + 20 + 2, waist_y + 2),
            (center_x - 20 + 2, waist_y + 2)
        ]
        draw.polygon(shadow_points, fill=shadow_color)
        
        # Hauptteil
        main_points = [
            (center_x - 25, shoulder_y),
            (center_x + 25, shoulder_y),
            (center_x + 20, waist_y),
            (center_x - 20, waist_y)
        ]
        draw.polygon(main_points, fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
        
        # Ärmel mit Style-Details
        if "vintage" in style.lower():
            # Puffärmel
            draw.ellipse([center_x - 35, shoulder_y, center_x - 15, shoulder_y + 30], 
                        fill=color + (255,), outline=self.darken_color(color) + (255,), width=1)
            draw.ellipse([center_x + 15, shoulder_y, center_x + 35, shoulder_y + 30], 
                        fill=color + (255,), outline=self.darken_color(color) + (255,), width=1)
        else:
            # Normale Ärmel
            left_sleeve = [
                (center_x - 25, shoulder_y),
                (center_x - 40, shoulder_y + 20),
                (center_x - 35, shoulder_y + 50),
                (center_x - 25, shoulder_y + 35)
            ]
            draw.polygon(left_sleeve, fill=color + (255,), outline=self.darken_color(color) + (255,), width=1)
            
            right_sleeve = [
                (center_x + 25, shoulder_y),
                (center_x + 40, shoulder_y + 20),
                (center_x + 35, shoulder_y + 50),
                (center_x + 25, shoulder_y + 35)
            ]
            draw.polygon(right_sleeve, fill=color + (255,), outline=self.darken_color(color) + (255,), width=1)
        
        # Details
        # Nähte
        draw.line([(center_x - 22, shoulder_y + 5), (center_x - 18, waist_y - 5)], 
                 fill=self.darken_color(color) + (255,), width=1)
        draw.line([(center_x + 22, shoulder_y + 5), (center_x + 18, waist_y - 5)], 
                 fill=self.darken_color(color) + (255,), width=1)
        
        # Optional: Muster
        if "modern" in style.lower():
            # Moderne Akzente
            draw.line([(center_x - 15, shoulder_y + 10), (center_x + 15, shoulder_y + 10)], 
                     fill=self.lighten_color(color) + (255,), width=2)
    
    def draw_premium_dress(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], style: str):
        """Premium Kleid mit Style-Details"""
        shoulder_y = 95
        waist_y = 235
        hem_y = 400
        
        # Schatten
        shadow_color = tuple(max(0, c - 40) for c in color) + (120,)
        shadow_points = [
            (center_x - 25 + 2, shoulder_y + 2),
            (center_x + 25 + 2, shoulder_y + 2),
            (center_x + 20 + 2, waist_y + 2),
            (center_x + 35 + 2, hem_y + 2),
            (center_x - 35 + 2, hem_y + 2),
            (center_x - 20 + 2, waist_y + 2)
        ]
        draw.polygon(shadow_points, fill=shadow_color)
        
        # Hauptkleid
        dress_points = [
            (center_x - 25, shoulder_y),
            (center_x + 25, shoulder_y),
            (center_x + 20, waist_y),
            (center_x + 35, hem_y),
            (center_x - 35, hem_y),
            (center_x - 20, waist_y)
        ]
        draw.polygon(dress_points, fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
        
        # Style-spezifische Details
        if "vintage" in style.lower():
            # Vintage-Gürtel
            draw.rectangle([center_x - 22, waist_y - 5, center_x + 22, waist_y + 5], 
                         fill=self.darken_color(color) + (255,))
            # Vintage-Knöpfe
            for i in range(3):
                button_y = shoulder_y + 15 + i * 20
                draw.ellipse([center_x - 3, button_y, center_x + 3, button_y + 6], 
                           fill=self.darken_color(color) + (255,))
        elif "modern" in style.lower():
            # Moderne asymmetrische Linie
            draw.line([(center_x - 20, waist_y), (center_x + 30, hem_y - 20)], 
                     fill=self.lighten_color(color) + (255,), width=3)
    
    def draw_premium_pants(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], style: str):
        """Premium Hose mit Details"""
        waist_y = 235
        crotch_y = 305
        knee_y = 415
        ankle_y = 520
        
        # Schatten
        shadow_color = tuple(max(0, c - 40) for c in color) + (120,)
        
        # Linkes Bein Schatten
        left_shadow = [
            (center_x - 20 + 2, waist_y + 2),
            (center_x - 2 + 2, waist_y + 2),
            (center_x - 2 + 2, crotch_y + 2),
            (center_x - 10 + 2, knee_y + 2),
            (center_x - 15 + 2, ankle_y + 2),
            (center_x - 28 + 2, ankle_y + 2),
            (center_x - 23 + 2, knee_y + 2),
            (center_x - 20 + 2, crotch_y + 2)
        ]
        draw.polygon(left_shadow, fill=shadow_color)
        
        # Rechtes Bein Schatten
        right_shadow = [
            (center_x + 2 + 2, waist_y + 2),
            (center_x + 20 + 2, waist_y + 2),
            (center_x + 20 + 2, crotch_y + 2),
            (center_x + 23 + 2, knee_y + 2),
            (center_x + 28 + 2, ankle_y + 2),
            (center_x + 15 + 2, ankle_y + 2),
            (center_x + 10 + 2, knee_y + 2),
            (center_x + 2 + 2, crotch_y + 2)
        ]
        draw.polygon(right_shadow, fill=shadow_color)
        
        # Haupthose
        # Linkes Bein
        left_leg = [
            (center_x - 20, waist_y),
            (center_x - 2, waist_y),
            (center_x - 2, crotch_y),
            (center_x - 10, knee_y),
            (center_x - 15, ankle_y),
            (center_x - 28, ankle_y),
            (center_x - 23, knee_y),
            (center_x - 20, crotch_y)
        ]
        draw.polygon(left_leg, fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
        
        # Rechtes Bein
        right_leg = [
            (center_x + 2, waist_y),
            (center_x + 20, waist_y),
            (center_x + 20, crotch_y),
            (center_x + 23, knee_y),
            (center_x + 28, ankle_y),
            (center_x + 15, ankle_y),
            (center_x + 10, knee_y),
            (center_x + 2, crotch_y)
        ]
        draw.polygon(right_leg, fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
        
        # Style-Details
        if "vintage" in style.lower():
            # Vintage Bügelfalte
            draw.line([(center_x - 11, waist_y + 10), (center_x - 11, ankle_y - 5)], 
                     fill=self.darken_color(color) + (255,), width=2)
            draw.line([(center_x + 11, waist_y + 10), (center_x + 11, ankle_y - 5)], 
                     fill=self.darken_color(color) + (255,), width=2)
        elif "modern" in style.lower():
            # Moderne Seitennähte
            draw.line([(center_x - 18, waist_y + 5), (center_x - 26, ankle_y - 5)], 
                     fill=self.lighten_color(color) + (255,), width=1)
            draw.line([(center_x + 18, waist_y + 5), (center_x + 26, ankle_y - 5)], 
                     fill=self.lighten_color(color) + (255,), width=1)
        
        # Mittelnaht
        draw.line([(center_x, waist_y), (center_x, crotch_y)], 
                 fill=self.darken_color(color) + (255,), width=2)
    
    def draw_premium_coat(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], style: str):
        """Premium Mantel mit Details"""
        shoulder_y = 85
        waist_y = 235
        hem_y = 420
        
        # Schatten
        shadow_color = tuple(max(0, c - 40) for c in color) + (120,)
        shadow_points = [
            (center_x - 32 + 3, shoulder_y + 3),
            (center_x + 32 + 3, shoulder_y + 3),
            (center_x + 27 + 3, waist_y + 3),
            (center_x + 45 + 3, hem_y + 3),
            (center_x - 45 + 3, hem_y + 3),
            (center_x - 27 + 3, waist_y + 3)
        ]
        draw.polygon(shadow_points, fill=shadow_color)
        
        # Hauptmantel
        coat_points = [
            (center_x - 32, shoulder_y),
            (center_x + 32, shoulder_y),
            (center_x + 27, waist_y),
            (center_x + 45, hem_y),
            (center_x - 45, hem_y),
            (center_x - 27, waist_y)
        ]
        draw.polygon(coat_points, fill=color + (255,), outline=self.darken_color(color) + (255,), width=3)
        
        # Ärmel
        left_sleeve = [
            (center_x - 32, shoulder_y),
            (center_x - 55, shoulder_y + 25),
            (center_x - 50, shoulder_y + 85),
            (center_x - 32, shoulder_y + 65)
        ]
        draw.polygon(left_sleeve, fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
        
        right_sleeve = [
            (center_x + 32, shoulder_y),
            (center_x + 55, shoulder_y + 25),
            (center_x + 50, shoulder_y + 85),
            (center_x + 32, shoulder_y + 65)
        ]
        draw.polygon(right_sleeve, fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
        
        # Premium Details
        # Knöpfe
        for i in range(4):
            button_y = shoulder_y + 40 + i * 30
            draw.ellipse([center_x - 4, button_y, center_x + 4, button_y + 8], 
                        fill=self.darken_color(color) + (255,))
        
        # Kragen
        collar_points = [
            (center_x - 12, shoulder_y - 5),
            (center_x + 12, shoulder_y - 5),
            (center_x + 18, shoulder_y + 12),
            (center_x - 18, shoulder_y + 12)
        ]
        draw.polygon(collar_points, fill=self.darken_color(color) + (255,))
        
        # Style-spezifische Details
        if "vintage" in style.lower():
            # Vintage Taschen
            draw.rectangle([center_x - 38, waist_y + 20, center_x - 25, waist_y + 35], 
                         fill=self.darken_color(color) + (255,))
            draw.rectangle([center_x + 25, waist_y + 20, center_x + 38, waist_y + 35], 
                         fill=self.darken_color(color) + (255,))
    
    def draw_premium_shoes(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], 
                          shoe_type: str, style: str):
        """Premium Schuhe mit Details"""
        ankle_y = 520
        
        if shoe_type == "Stiefeletten":
            boot_height = 50
            
            # Schatten
            shadow_color = tuple(max(0, c - 50) for c in color) + (120,)
            draw.polygon([
                (center_x - 25 + 2, ankle_y - boot_height + 2),
                (center_x - 10 + 2, ankle_y - boot_height + 2),
                (center_x - 6 + 2, ankle_y + 2),
                (center_x - 30 + 2, ankle_y + 2),
                (center_x - 32 + 2, ankle_y - 12 + 2)
            ], fill=shadow_color)
            
            draw.polygon([
                (center_x + 10 + 2, ankle_y - boot_height + 2),
                (center_x + 25 + 2, ankle_y - boot_height + 2),
                (center_x + 32 + 2, ankle_y - 12 + 2),
                (center_x + 30 + 2, ankle_y + 2),
                (center_x + 6 + 2, ankle_y + 2)
            ], fill=shadow_color)
            
            # Linker Stiefel
            draw.polygon([
                (center_x - 25, ankle_y - boot_height),
                (center_x - 10, ankle_y - boot_height),
                (center_x - 6, ankle_y),
                (center_x - 30, ankle_y),
                (center_x - 32, ankle_y - 12)
            ], fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
            
            # Rechter Stiefel
            draw.polygon([
                (center_x + 10, ankle_y - boot_height),
                (center_x + 25, ankle_y - boot_height),
                (center_x + 32, ankle_y - 12),
                (center_x + 30, ankle_y),
                (center_x + 6, ankle_y)
            ], fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
            
            # Stiefel-Details
            # Schnürung
            for i in range(3):
                detail_y = ankle_y - boot_height + 12 + i * 12
                draw.line([(center_x - 22, detail_y), (center_x - 13, detail_y)], 
                         fill=self.darken_color(color) + (255,), width=1)
                draw.line([(center_x + 13, detail_y), (center_x + 22, detail_y)], 
                         fill=self.darken_color(color) + (255,), width=1)
                # Ösen
                draw.ellipse([center_x - 23, detail_y - 1, center_x - 21, detail_y + 1], 
                           fill=self.lighten_color(color) + (255,))
                draw.ellipse([center_x + 21, detail_y - 1, center_x + 23, detail_y + 1], 
                           fill=self.lighten_color(color) + (255,))
        
        else:
            # Normale Schuhe mit Premium-Details
            # Schatten
            shadow_color = tuple(max(0, c - 50) for c in color) + (120,)
            draw.ellipse([center_x - 25 + 2, ankle_y - 10 + 2, center_x - 2 + 2, ankle_y + 10 + 2], 
                        fill=shadow_color)
            draw.ellipse([center_x + 2 + 2, ankle_y - 10 + 2, center_x + 25 + 2, ankle_y + 10 + 2], 
                        fill=shadow_color)
            
            # Linker Schuh
            draw.ellipse([center_x - 25, ankle_y - 10, center_x - 2, ankle_y + 10], 
                        fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
            # Rechter Schuh
            draw.ellipse([center_x + 2, ankle_y - 10, center_x + 25, ankle_y + 10], 
                        fill=color + (255,), outline=self.darken_color(color) + (255,), width=2)
            
            # Schuh-Details
            if shoe_type == "Sneaker":
                # Sneaker-Sohle
                draw.arc([center_x - 24, ankle_y - 8, center_x - 3, ankle_y + 8], 0, 180, 
                        fill=self.lighten_color(color) + (255,), width=2)
                draw.arc([center_x + 3, ankle_y - 8, center_x + 24, ankle_y + 8], 0, 180, 
                        fill=self.lighten_color(color) + (255,), width=2)
                # Logo
                draw.rectangle([center_x - 20, ankle_y - 5, center_x - 15, ankle_y], 
                             fill=self.darken_color(color) + (255,))
                draw.rectangle([center_x + 15, ankle_y - 5, center_x + 20, ankle_y], 
                             fill=self.darken_color(color) + (255,))
    
    def add_professional_effects(self, img: Image.Image) -> Image.Image:
        """Fügt professionelle Effekte hinzu"""
        # Subtile Schatten
        shadow_img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Beleuchtungs-Overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = img.size
        center_x, center_y = width // 2, height // 3
        
        # Hauptlicht (von oben links)
        for radius in range(120, 0, -6):
            alpha = int(8 * (120 - radius) / 120)
            draw.ellipse([
                center_x - 60 - radius//2, center_y - 60 - radius//2,
                center_x - 60 + radius//2, center_y - 60 + radius//2
            ], fill=(255, 255, 255, alpha))
        
        # Fülllicht (von rechts)
        for radius in range(100, 0, -8):
            alpha = int(6 * (100 - radius) / 100)
            draw.ellipse([
                center_x + 50 - radius//2, center_y - radius//2,
                center_x + 50 + radius//2, center_y + radius//2
            ], fill=(255, 245, 220, alpha))
        
        # Composite
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        return img
    
    def enhance_image_quality(self, img: Image.Image) -> Image.Image:
        """Verbessert Bildqualität"""
        # Schärfe
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.15)
        
        # Kontrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.08)
        
        # Sättigung
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.05)
        
        return img
    
    def get_premium_colors(self, style_prompt: str) -> List[Tuple[int, int, int]]:
        """Premium Farben basierend auf Style"""
        style_lower = style_prompt.lower()
        
        if "modern" in style_lower:
            return [(45, 45, 48), (240, 240, 245), (128, 128, 130), (220, 220, 225)]
        elif "klassisch" in style_lower:
            return [(25, 25, 112), (255, 255, 255), (139, 69, 19), (105, 105, 105)]
        elif "vintage" in style_lower:
            return [(139, 69, 19), (205, 133, 63), (128, 0, 0), (160, 82, 45)]
        elif "sportlich" in style_lower:
            return [(0, 0, 0), (255, 69, 0), (255, 255, 255), (30, 144, 255)]
        elif "extravagant" in style_lower:
            return [(148, 0, 211), (255, 20, 147), (255, 215, 0), (50, 205, 50)]
        else:
            return [(47, 79, 79), (205, 192, 176), (139, 69, 19), (112, 128, 144)]
    
    def darken_color(self, color: Tuple[int, int, int], factor: float = 0.6) -> Tuple[int, int, int]:
        """Dunkelt Farbe ab"""
        return tuple(max(0, int(c * factor)) for c in color[:3])
    
    def lighten_color(self, color: Tuple[int, int, int], factor: float = 1.4) -> Tuple[int, int, int]:
        """Hellt Farbe auf"""
        return tuple(min(255, int(c * factor)) for c in color[:3])

def create_professional_fashion_prompt(selected_items: List[Dict], style_prompt: str) -> str:
    """Erstellt professionelle Fashion-Photography Prompts"""
    categories = [item['category'] for item in selected_items]
    category_text = ", ".join(set(categories))
    
    base_prompt = f"professional fashion photography, elegant female model wearing {category_text}"
    
    style_additions = {
        "modern": "contemporary minimalist design, clean lines, urban chic, sleek modern aesthetic",
        "klassisch": "timeless elegant style, refined sophisticated look, classic tailoring",
        "vintage": "retro inspired fashion, classic vintage aesthetic, nostalgic charm",
        "sportlich": "athletic wear, dynamic pose, fitness lifestyle, activewear elegance",
        "extravagant": "high fashion couture, dramatic styling, avant-garde design",
        "minimalistisch": "clean minimal aesthetic, neutral colors, simple elegance",
        "natürlich": "organic natural materials, earth tones, relaxed fit, sustainable fashion",
        "monochrom": "black and white styling, monochromatic palette, tonal harmony",
        "pastell": "soft pastel colors, dreamy romantic style, delicate hues"
    }
    
    style_addition = ""
    for key, addition in style_additions.items():
        if key in style_prompt.lower():
            style_addition = addition
            break
    
    if not style_addition:
        style_addition = "contemporary fashion styling, modern elegance"
    
    full_prompt = f"{base_prompt}, {style_addition}, professional studio lighting, high fashion photography, editorial quality, runway model pose, detailed fabric textures, photorealistic, 8k resolution, magazine cover quality, vogue style"
    
    return full_prompt

def create_sample_fashion_items():
    """Erstellt Sample Fashion Items"""
    items = []
    brands = ["Chanel", "Dior", "Versace", "Prada", "Gucci", "Armani", "Zara", "H&M", "COS", "Uniqlo"]
    prices = ["45€", "65€", "85€", "120€", "150€", "200€", "250€", "300€", "450€", "600€"]
    
    for i in range(20):
        category_idx = i % len(FASHION_CLASSES)
        category = FASHION_CLASSES[category_idx]
        
        # Erstelle bessere Sample-Bilder
        img = Image.new('RGB', (28, 28), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Verschiedene Formen je nach Kategorie
        if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
            draw.rectangle([4, 6, 24, 18], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
            draw.rectangle([2, 8, 6, 16], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
            draw.rectangle([22, 8, 26, 16], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
        elif category == "Kleid":
            draw.polygon([(6, 6), (22, 6), (24, 26), (4, 26)], fill=(random.randint(100, 255), random.randint(50, 150), random.randint(100, 255)))
        elif category == "Hose":
            draw.rectangle([6, 10, 10, 26], fill=(random.randint(20, 100), random.randint(20, 100), random.randint(50, 150)))
            draw.rectangle([18, 10, 22, 26], fill=(random.randint(20, 100), random.randint(20, 100), random.randint(50, 150)))
        elif category == "Mantel":
            draw.rectangle([2, 4, 26, 26], fill=(random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)))
            draw.rectangle([0, 6, 4, 20], fill=(random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)))
            draw.rectangle([24, 6, 28, 20], fill=(random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)))
        elif category in ["Sneaker", "Sandalen", "Stiefeletten"]:
            draw.ellipse([4, 18, 12, 26], fill=(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
            draw.ellipse([16, 18, 24, 26], fill=(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
        elif category == "Tasche":
            draw.rectangle([8, 8, 20, 20], fill=(random.randint(100, 200), random.randint(50, 150), random.randint(50, 150)))
            draw.line([(10, 8), (18, 8)], fill=(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)), width=2)
        else:
            draw.ellipse([6, 6, 22, 22], fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
        
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

def generate_fashion_design_premium(selected_items: List[Dict], style_prompt: str, 
                                  generator: PremiumFashionGenerator) -> Optional[np.ndarray]:
    """Generiert Premium Fashion-Designs"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🎨 Erstelle professionellen Fashion-Prompt...")
        progress_bar.progress(0.1)
        
        prompt = create_professional_fashion_prompt(selected_items, style_prompt)
        
        st.write(f"**🎭 Generierungs-Prompt:** {prompt[:120]}...")
        
        # API Verfügbarkeit prüfen
        available_apis = generator.check_api_availability()
        generated_image = None
        
        # Versuche APIs in Reihenfolge der Qualität
        if available_apis['stability']:
            status_text.text("🎯 Generiere mit Stability AI (Premium Quality)...")
            progress_bar.progress(0.3)
            generated_image = generator.generate_with_stability_api(prompt)
            if generated_image:
                status_text.text("✅ Stability AI erfolgreich!")
        
        if generated_image is None and available_apis['openai']:
            status_text.text("🤖 Versuche OpenAI DALL-E 3...")
            progress_bar.progress(0.5)
            generated_image = generator.generate_with_openai_dalle(prompt)
            if generated_image:
                status_text.text("✅ DALL-E 3 erfolgreich!")
        
        if generated_image is None and available_apis['replicate']:
            status_text.text("🔄 Versuche Replicate SDXL...")
            progress_bar.progress(0.7)
            generated_image = generator.generate_with_replicate_api(prompt)
            if generated_image:
                status_text.text("✅ Replicate erfolgreich!")
        
        if generated_image is None and available_apis['huggingface']:
            status_text.text("🤗 Versuche Hugging Face API...")
            progress_bar.progress(0.8)
            generated_image = generator.generate_with_huggingface_api(prompt)
            if generated_image:
                status_text.text("✅ Hugging Face erfolgreich!")
        
        if generated_image is None:
            # Enhanced Fallback
            status_text.text("🎨 Erstelle Premium Fallback-Design...")
            progress_bar.progress(0.9)
            generated_image = generator.create_enhanced_fallback_design(selected_items, style_prompt)
            status_text.text("✅ Premium Fallback-Design erstellt!")
        
        progress_bar.progress(1.0)
        
        return np.array(generated_image).astype(np.float32) / 255.0
        
    except Exception as e:
        st.error(f"❌ Fehler bei der Generierung: {e}")
        # Notfall-Fallback
        try:
            status_text.text("🎨 Erstelle Notfall-Design...")
            fallback_image = generator.create_enhanced_fallback_design(selected_items, style_prompt)
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
        st.session_state.generator = PremiumFashionGenerator()

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
    st.markdown("## 🎨 Premium Fashion-Generator")
    
    # API Status anzeigen
    generator = st.session_state.generator
    available_apis = generator.check_api_availability()
    
    if any(available_apis.values()):
        api_list = []
        if available_apis['stability']:
            api_list.append("✅ **Stability AI** (Premium SDXL)")
        if available_apis['openai']:
            api_list.append("✅ **OpenAI DALL-E 3** (Premium)")
        if available_apis['replicate']:
            api_list.append("✅ **Replicate SDXL** (Hochqualität)")
        if available_apis['huggingface']:
            api_list.append("✅ **Hugging Face** (Kostenlos)")
        
        st.markdown(f"""
        <div class="generation-alert">
            <h3>🚀 Premium AI-APIs verfügbar!</h3>
            <p>Folgende AI-Services sind konfiguriert und erstellen professionelle Fashion-Fotografien:</p>
            <ul style="margin: 10px 0;">
                {"".join([f"<li>{api}</li>" for api in api_list])}
            </ul>
            <p><strong>💎 Ergebnis:</strong> Realistische, hochauflösende Fashion-Designs!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="generation-alert">
            <h3>💡 Setup für Premium AI-Features</h3>
            <p>Für professionelle AI-generierte Fashion-Fotos, füge API-Keys in den Streamlit Secrets hinzu:</p>
            <details>
                <summary><strong>🔑 API-Setup Anleitung</strong></summary>
                <br>
                <p><strong>In Streamlit Cloud → Settings → Secrets:</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
# Beste Qualität (empfohlen)
STABILITY_API_KEY = "sk-..."

# Alternative Premium Option  
OPENAI_API_KEY = "sk-..."

# Gute kostenlose Option
HUGGINGFACE_TOKEN = "hf_..."

# Weitere Alternative
REPLICATE_API_TOKEN = "r8_..."</pre>
                <p><strong>🎯 Empfehlung:</strong> Stability AI für beste Fashion-Fotografie Qualität!</p>
                <p><strong>💰 Kosten:</strong></p>
                <ul>
                    <li>Stability AI: ~$0.04 pro Bild</li>
                    <li>OpenAI DALL-E 3: ~$0.08 pro Bild</li>
                    <li>Hugging Face: Kostenlos (langsamer)</li>
                </ul>
            </details>
            <p><strong>Ohne APIs:</strong> Verwendet optimierten lokalen Generator</p>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.all_time_favorites:
        st.warning("💡 Sammle erst Favoriten durch Swipen in der Fashion-Kollektion!")
        return
    
    # Favoriten-Auswahl
    st.markdown("### 👗 Wähle Fashion-Styles (max. 3 für beste Ergebnisse)")
    
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
                        st.warning("⚠️ Maximal 3 Styles für optimale Ergebnisse!")
                st.rerun()
            
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="{item['image_data']}" 
                     style="width: 100%; border-radius: 8px; margin-bottom: 5px;
                            border: {'3px solid #4CAF50' if is_selected else '1px solid #ddd'};">
                <div style="font-size: 0.8rem; color: #666;">{item['category']}</div>
                <div style="font-size: 0.7rem; color: #999;">{item['brand']}</div>
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
        
        col1, col2, col3 = st.columns(3)
        
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
        
        with col3:
            photo_style = st.selectbox(
                "📸 Foto-Stil",
                ["Studio", "Editorial", "Runway", "Street Style", "Commercial"],
                help="Art der Fashion-Fotografie"
            )
        
        style_prompt = f"{style_mood.lower()} {photo_style.lower()} style, {color_scheme.lower()} colors"
        
        # Generate-Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎨 ✨ Generiere Premium Fashion-Foto!", type="primary", use_container_width=True):
                with st.spinner("Erstelle professionelles Fashion-Design..."):
                    generated_image = generate_fashion_design_premium(
                        st.session_state.selected_for_generation,
                        style_prompt,
                        st.session_state.generator
                    )
                    
                    if generated_image is not None:
                        # Speichere Generierung
                        generation_data = {
                            'image': numpy_to_base64(generated_image, size=(512, 768)),
                            'style': f"{style_mood} {photo_style}",
                            'colors': color_scheme,
                            'items': [item['category'] for item in st.session_state.selected_for_generation],
                            'timestamp': datetime.now().isoformat(),
                            'api_used': 'Premium AI' if any(st.session_state.generator.check_api_availability().values()) else 'Enhanced Local'
                        }
                        
                        st.session_state.generated_images.append(generation_data)
                        if len(st.session_state.generated_images) > 15:
                            st.session_state.generated_images.pop(0)
                        
                        # Darstellung des Ergebnisses
                        st.markdown("---")
                        st.markdown("## 📸 Generiertes Premium Fashion-Design")
                        
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("""
                            <div class="runway-preview">
                                <div style="position: relative; z-index: 1;">
                            """, unsafe_allow_html=True)
                            
                            st.image(generated_image, 
                                   caption=f"Premium Fashion-Design ({generation_data['api_used']})",
                                   use_column_width=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: white; padding: 20px; border-radius: 15px; 
                                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                                <h4>📋 Design-Details</h4>
                                <p><strong>Style:</strong> {style_mood} {photo_style}</p>
                                <p><strong>Farben:</strong> {color_scheme}</p>
                                <p><strong>Fashion-Items:</strong><br>
                                   {', '.join([item['category'] for item in st.session_state.selected_for_generation])}</p>
                                <p><strong>Generiert mit:</strong> {generation_data['api_used']}</p>
                                <p><strong>Zeit:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Download-Button
                            img_buffer = io.BytesIO()
                            img = Image.fromarray((generated_image * 255).astype(np.uint8))
                            img.save(img_buffer, format="PNG", optimize=True, quality=95)
                            img_buffer.seek(0)
                            
                            st.download_button(
                                "💾 Premium Design herunterladen",
                                data=img_buffer,
                                file_name=f"premium_fashion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ Fehler bei der Generierung. Bitte versuche es erneut oder überprüfe die API-Konfiguration.")

def render_gallery_tab():
    """Zeigt Premium Galerie"""
    st.markdown("## 🖼️ Premium Fashion-Design Galerie")
    
    if not st.session_state.generated_images:
        st.markdown("""
        <div style="text-align: center; padding: 60px; color: #666;">
            <h3>📷 Noch keine Premium Fashion-Designs generiert</h3>
            <p>Gehe zum Generator-Tab und erstelle dein erstes professionelles Fashion-Design!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"**📸 {len(st.session_state.generated_images)} Premium Fashion-Designs generiert**")
    
    # Galerie mit verbesserter Darstellung
    for idx, gen_data in enumerate(reversed(st.session_state.generated_images)):
        with st.container():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.image(gen_data['image'], use_column_width=True)
            
            with col2:
                api_badge = "🎯 Premium AI" if "Premium" in gen_data.get('api_used', '') else "🎨 Enhanced Local"
                
                st.markdown(f"""
                <div style="padding: 20px;">
                    <h4>📸 Fashion-Design #{len(st.session_state.generated_images) - idx}</h4>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <strong>{api_badge}</strong>
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
    <div class="main-header">🏃‍♀️ Fashion Swipe Studio Pro</div>
    <div class="sub-header">Premium AI Fashion Generator</div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔄 Fashion Swipe", "🎨 Premium Generator", "🖼️ Premium Galerie"])
    
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
        
        st.metric("📸 Premium Designs", len(st.session_state.generated_images))
        
        st.markdown("---")
        
        # Action Buttons
        if st.button("🔄 Neue Session", use_container_width=True):
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
        
        for api_name, is_available in available_apis.items():
            if is_available:
                st.markdown(f"""
                <div class="api-status">
                    <strong>✅ {api_name.title()}</strong><br>
                    <small>Verfügbar</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="api-status unavailable">
                    <strong>❌ {api_name.title()}</strong><br>
                    <small>Nicht konfiguriert</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; font-size: 0.8rem;">
            <strong>💎 Fashion Swipe Studio Pro:</strong><br>
            • Premium AI-Integration<br>
            • Realistische Fashion-Fotos<br>
            • Professionelle Qualität<br>
            • Mehrere AI-Services<br>
            • Enhanced Fallback-Modus
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
