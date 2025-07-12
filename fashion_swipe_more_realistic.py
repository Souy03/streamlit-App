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
import torch
from collections import Counter
import gc
import os
import requests
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Realistic Fashion Generator",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Memory-optimierte Konfiguration
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

# Custom CSS (erweitert für professionelle Darstellung)
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
        position: relative;
        cursor: grab;
        user-select: none;
        touch-action: none;
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
    
    .runway-preview::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="%23333" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .generated-image {
        border-radius: 15px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .generated-image:hover {
        transform: scale(1.02);
    }
    
    .generation-controls {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 30px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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

# Professionelle Fashion-Generator Klasse
class RealisticFashionGenerator:
    """Generiert realistische Fashion-Fotografien"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_model = None
        
    def generate_with_sdxl_turbo(self, prompt: str, negative_prompt: str = "") -> Optional[Image.Image]:
        """Nutze SDXL-Turbo für realistische Fashion-Fotografien"""
        try:
            from diffusers import AutoPipelineForText2Image
            import torch
            
            # Lade Modell nur wenn nötig
            if self.current_model != 'sdxl-turbo':
                self.cleanup_memory()
                
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                    use_safetensors=True
                )
                pipe = pipe.to(self.device)
                
                # Memory-Optimierungen
                if torch.cuda.is_available():
                    pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                
                self.pipe = pipe
                self.current_model = 'sdxl-turbo'
            
            # Generiere hochqualitatives Bild
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=2,  # Turbo braucht nur 1-4 Steps
                guidance_scale=0.0,
                height=768,  # Höhere Auflösung für bessere Qualität
                width=512
            ).images[0]
            
            # Verbessere Bildqualität
            image = self.enhance_image_quality(image)
            
            return image
            
        except ImportError as e:
            st.error(f"Diffusers Import-Fehler: {e}")
            st.info("Installiere: pip install --upgrade diffusers transformers accelerate")
            return None
        except Exception as e:
            st.error(f"SDXL-Turbo Fehler: {e}")
            return None
    
    def generate_with_huggingface_api(self, prompt: str) -> Optional[Image.Image]:
        """Nutze Hugging Face Inference API für realistische Bilder"""
        try:
            # Verwende ein auf Fashion spezialisiertes Modell
            API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN', '')}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 768
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return self.enhance_image_quality(image)
            else:
                return None
                
        except Exception as e:
            st.warning(f"API nicht verfügbar: {e}")
            return None
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Verbessert die Bildqualität mit klassischen Methoden"""
        try:
            # Erhöhe Schärfe
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Verbessere Kontrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Erhöhe Sättigung leicht
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            
            # Skaliere auf finale Größe
            image = image.resize((768, 1024), Image.LANCZOS)
            
            return image
        except Exception:
            return image.resize((768, 1024), Image.LANCZOS)
    
    def create_professional_fashion_image(self, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Erstellt professionelle Fashion-Fotografien als Fallback"""
        
        # Hochauflösender Canvas
        img = Image.new('RGB', (768, 1024), color=(240, 240, 245))
        
        # Erstelle professionellen Runway-Hintergrund
        img = self.create_runway_background(img)
        
        # Lade oder erstelle realistisches Model-Base
        model_img = self.create_realistic_model_base(img.size)
        
        # Füge Kleidung basierend auf Auswahl hinzu
        model_img = self.apply_realistic_clothing(model_img, selected_items, style_prompt)
        
        # Füge professionelle Beleuchtung hinzu
        model_img = self.add_professional_lighting(model_img)
        
        # Composite auf Runway
        final_img = Image.alpha_composite(img.convert('RGBA'), model_img.convert('RGBA'))
        
        return final_img.convert('RGB')
    
    def create_runway_background(self, base_img: Image.Image) -> Image.Image:
        """Erstellt einen professionellen Runway-Hintergrund"""
        draw = ImageDraw.Draw(base_img)
        width, height = base_img.size
        
        # Gradient Hintergrund (dunkel für Runway)
        for y in range(height):
            # Dunkler Gradient von oben nach unten
            shade = int(20 + (y / height) * 40)  # Von sehr dunkel zu dunkel
            draw.rectangle([0, y, width, y+1], fill=(shade, shade, shade+5))
        
        # Runway-Laufsteg
        runway_width = width // 3
        runway_x = (width - runway_width) // 2
        
        # Laufsteg mit Perspektive
        for y in range(height//2, height):
            perspective = (y - height//2) / (height//2)
            current_width = int(runway_width * (0.3 + 0.7 * perspective))
            x_start = runway_x + (runway_width - current_width) // 2
            
            # Laufsteg-Farbe (hellgrau)
            gray = int(180 - perspective * 50)
            draw.rectangle([x_start, y, x_start + current_width, y+1], 
                          fill=(gray, gray, gray+10))
        
        # Spotlight-Effekte
        self.add_spotlight_effects(base_img)
        
        return base_img
    
    def add_spotlight_effects(self, img: Image.Image):
        """Fügt Spotlight-Beleuchtungseffekte hinzu"""
        # Erstelle Spotlight-Overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = img.size
        center_x, center_y = width // 2, height // 3
        
        # Hauptspotlight
        for radius in range(300, 0, -5):
            alpha = int(30 * (300 - radius) / 300)
            draw.ellipse([center_x - radius, center_y - radius//2, 
                         center_x + radius, center_y + radius//2], 
                        fill=(255, 255, 255, alpha))
        
        # Mische Overlay
        img.paste(overlay, (0, 0), overlay)
    
    def create_realistic_model_base(self, size: Tuple[int, int]) -> Image.Image:
        """Erstellt eine realistische Model-Basis-Silhouette"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Model-Proportionen (realistischer)
        center_x = width // 2
        head_y = height // 6
        shoulder_y = head_y + 80
        waist_y = shoulder_y + 180
        hip_y = waist_y + 120
        knee_y = hip_y + 200
        ankle_y = height - 80
        
        # Hautfarbe
        skin_color = (245, 222, 179, 255)  # Natürliche Hautfarbe
        
        # Kopf (oval, realistischer)
        head_width, head_height = 60, 80
        draw.ellipse([center_x - head_width//2, head_y - head_height//2,
                     center_x + head_width//2, head_y + head_height//2], 
                    fill=skin_color)
        
        # Hals
        draw.rectangle([center_x - 15, head_y + head_height//2,
                       center_x + 15, shoulder_y - 20], fill=skin_color)
        
        # Körper (realistischere Silhouette)
        # Torso
        shoulder_width = 80
        waist_width = 55
        hip_width = 75
        
        # Torso-Polygon für natürliche Form
        torso_points = [
            (center_x - shoulder_width//2, shoulder_y),
            (center_x + shoulder_width//2, shoulder_y),
            (center_x + waist_width//2, waist_y),
            (center_x + hip_width//2, hip_y),
            (center_x - hip_width//2, hip_y),
            (center_x - waist_width//2, waist_y)
        ]
        draw.polygon(torso_points, fill=skin_color)
        
        # Arme (realistischer)
        arm_width = 25
        # Linker Arm
        draw.polygon([
            (center_x - shoulder_width//2, shoulder_y),
            (center_x - shoulder_width//2 - 40, shoulder_y + 100),
            (center_x - shoulder_width//2 - 35, shoulder_y + 200),
            (center_x - shoulder_width//2 - 30, waist_y + 50),
            (center_x - shoulder_width//2 - arm_width, waist_y + 50),
            (center_x - shoulder_width//2 - arm_width - 5, shoulder_y + 200),
            (center_x - shoulder_width//2 - arm_width - 10, shoulder_y + 100),
            (center_x - shoulder_width//2 - arm_width, shoulder_y)
        ], fill=skin_color)
        
        # Rechter Arm
        draw.polygon([
            (center_x + shoulder_width//2, shoulder_y),
            (center_x + shoulder_width//2 + 40, shoulder_y + 100),
            (center_x + shoulder_width//2 + 35, shoulder_y + 200),
            (center_x + shoulder_width//2 + 30, waist_y + 50),
            (center_x + shoulder_width//2 + arm_width, waist_y + 50),
            (center_x + shoulder_width//2 + arm_width + 5, shoulder_y + 200),
            (center_x + shoulder_width//2 + arm_width + 10, shoulder_y + 100),
            (center_x + shoulder_width//2 + arm_width, shoulder_y)
        ], fill=skin_color)
        
        # Beine (realistischer)
        leg_width = 35
        # Linkes Bein
        draw.polygon([
            (center_x - hip_width//2, hip_y),
            (center_x - leg_width//2, hip_y),
            (center_x - leg_width//2, knee_y),
            (center_x - leg_width//2 - 5, ankle_y),
            (center_x - leg_width//2 - 25, ankle_y),
            (center_x - leg_width//2 - 20, knee_y),
            (center_x - hip_width//2, knee_y)
        ], fill=skin_color)
        
        # Rechtes Bein
        draw.polygon([
            (center_x + hip_width//2, hip_y),
            (center_x + leg_width//2, hip_y),
            (center_x + leg_width//2, knee_y),
            (center_x + leg_width//2 + 5, ankle_y),
            (center_x + leg_width//2 + 25, ankle_y),
            (center_x + leg_width//2 + 20, knee_y),
            (center_x + hip_width//2, knee_y)
        ], fill=skin_color)
        
        # Füge Gesichtszüge hinzu
        self.add_facial_features(img, center_x, head_y)
        
        # Füge Haare hinzu
        self.add_realistic_hair(img, center_x, head_y, head_width, head_height)
        
        return img
    
    def add_facial_features(self, img: Image.Image, center_x: int, head_y: int):
        """Fügt realistische Gesichtszüge hinzu"""
        draw = ImageDraw.Draw(img)
        
        # Augen
        eye_y = head_y - 10
        draw.ellipse([center_x - 25, eye_y - 5, center_x - 15, eye_y + 5], fill=(70, 50, 30))
        draw.ellipse([center_x + 15, eye_y - 5, center_x + 25, eye_y + 5], fill=(70, 50, 30))
        
        # Augenbrauen
        draw.arc([center_x - 28, eye_y - 15, center_x - 12, eye_y - 5], 0, 180, fill=(101, 67, 33), width=2)
        draw.arc([center_x + 12, eye_y - 15, center_x + 28, eye_y - 5], 0, 180, fill=(101, 67, 33), width=2)
        
        # Nase
        draw.polygon([(center_x - 3, head_y), (center_x + 3, head_y), (center_x, head_y + 15)], 
                    fill=(235, 200, 160))
        
        # Mund
        draw.ellipse([center_x - 8, head_y + 20, center_x + 8, head_y + 28], fill=(200, 120, 120))
    
    def add_realistic_hair(self, img: Image.Image, center_x: int, head_y: int, head_width: int, head_height: int):
        """Fügt realistische Haare hinzu"""
        draw = ImageDraw.Draw(img)
        hair_color = (101, 67, 33, 255)  # Braun
        
        # Haupthaar (oben und seitlich)
        hair_points = [
            (center_x - head_width//2 - 10, head_y - head_height//2 - 20),
            (center_x + head_width//2 + 10, head_y - head_height//2 - 20),
            (center_x + head_width//2 + 15, head_y - head_height//2 + 30),
            (center_x + head_width//2, head_y + head_height//2),
            (center_x - head_width//2, head_y + head_height//2),
            (center_x - head_width//2 - 15, head_y - head_height//2 + 30)
        ]
        draw.polygon(hair_points, fill=hair_color)
    
    def apply_realistic_clothing(self, model_img: Image.Image, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Wendet realistische Kleidung auf das Model an"""
        draw = ImageDraw.Draw(model_img)
        width, height = model_img.size
        center_x = width // 2
        
        # Bestimme Kleidungsfarben basierend auf Style
        clothing_colors = self.get_clothing_colors_from_style(style_prompt)
        
        categories = [item['category'] for item in selected_items]
        
        for i, category in enumerate(set(categories)):
            color = clothing_colors[i % len(clothing_colors)]
            
            if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
                self.draw_realistic_top(draw, center_x, color)
            elif category == "Kleid":
                self.draw_realistic_dress(draw, center_x, color)
            elif category == "Hose":
                self.draw_realistic_pants(draw, center_x, color)
            elif category == "Mantel":
                self.draw_realistic_coat(draw, center_x, color)
            elif category in ["Sneaker", "Sandalen", "Stiefeletten"]:
                self.draw_realistic_shoes(draw, center_x, color, category)
        
        # Füge Schatten und Details hinzu
        model_img = self.add_clothing_details(model_img)
        
        return model_img
    
    def draw_realistic_top(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet ein realistisches Oberteil"""
        # Oberteil mit realistischen Proportionen
        shoulder_y = 150
        waist_y = 330
        
        # Hauptteil
        top_points = [
            (center_x - 40, shoulder_y),
            (center_x + 40, shoulder_y),
            (center_x + 35, waist_y),
            (center_x - 35, waist_y)
        ]
        draw.polygon(top_points, fill=color, outline=self.darken_color(color), width=2)
        
        # Ärmel
        draw.polygon([
            (center_x - 40, shoulder_y),
            (center_x - 65, shoulder_y + 30),
            (center_x - 60, shoulder_y + 80),
            (center_x - 40, shoulder_y + 60)
        ], fill=color, outline=self.darken_color(color), width=1)
        
        draw.polygon([
            (center_x + 40, shoulder_y),
            (center_x + 65, shoulder_y + 30),
            (center_x + 60, shoulder_y + 80),
            (center_x + 40, shoulder_y + 60)
        ], fill=color, outline=self.darken_color(color), width=1)
    
    def draw_realistic_dress(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet ein realistisches Kleid"""
        shoulder_y = 150
        waist_y = 330
        hem_y = 650
        
        # Kleid-Silhouette
        dress_points = [
            (center_x - 40, shoulder_y),
            (center_x + 40, shoulder_y),
            (center_x + 35, waist_y),
            (center_x + 60, hem_y),
            (center_x - 60, hem_y),
            (center_x - 35, waist_y)
        ]
        draw.polygon(dress_points, fill=color, outline=self.darken_color(color), width=2)
        
        # Ärmel
        self.draw_dress_sleeves(draw, center_x, shoulder_y, color)
    
    def draw_dress_sleeves(self, draw: ImageDraw.Draw, center_x: int, shoulder_y: int, color: Tuple[int, int, int]):
        """Zeichnet Kleid-Ärmel"""
        # Kurze Ärmel
        draw.polygon([
            (center_x - 40, shoulder_y),
            (center_x - 55, shoulder_y + 20),
            (center_x - 50, shoulder_y + 40),
            (center_x - 40, shoulder_y + 30)
        ], fill=color, outline=self.darken_color(color), width=1)
        
        draw.polygon([
            (center_x + 40, shoulder_y),
            (center_x + 55, shoulder_y + 20),
            (center_x + 50, shoulder_y + 40),
            (center_x + 40, shoulder_y + 30)
        ], fill=color, outline=self.darken_color(color), width=1)
    
    def draw_realistic_pants(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet realistische Hosen"""
        waist_y = 330
        crotch_y = 450
        knee_y = 600
        ankle_y = 750
        
        # Linkes Bein
        draw.polygon([
            (center_x - 35, waist_y),
            (center_x - 5, waist_y),
            (center_x - 5, crotch_y),
            (center_x - 15, knee_y),
            (center_x - 20, ankle_y),
            (center_x - 40, ankle_y),
            (center_x - 35, knee_y),
            (center_x - 35, crotch_y)
        ], fill=color, outline=self.darken_color(color), width=2)
        
        # Rechtes Bein
        draw.polygon([
            (center_x + 5, waist_y),
            (center_x + 35, waist_y),
            (center_x + 35, crotch_y),
            (center_x + 35, knee_y),
            (center_x + 40, ankle_y),
            (center_x + 20, ankle_y),
            (center_x + 15, knee_y),
            (center_x + 5, crotch_y)
        ], fill=color, outline=self.darken_color(color), width=2)
    
    def draw_realistic_coat(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int]):
        """Zeichnet einen realistischen Mantel"""
        shoulder_y = 140
        waist_y = 330
        hem_y = 600
        
        # Mantel-Silhouette (weiter und länger)
        coat_points = [
            (center_x - 50, shoulder_y),
            (center_x + 50, shoulder_y),
            (center_x + 45, waist_y),
            (center_x + 70, hem_y),
            (center_x - 70, hem_y),
            (center_x - 45, waist_y)
        ]
        draw.polygon(coat_points, fill=color, outline=self.darken_color(color), width=3)
        
        # Mantel-Ärmel (länger)
        draw.polygon([
            (center_x - 50, shoulder_y),
            (center_x - 80, shoulder_y + 40),
            (center_x - 75, shoulder_y + 120),
            (center_x - 50, shoulder_y + 100)
        ], fill=color, outline=self.darken_color(color), width=2)
        
        draw.polygon([
            (center_x + 50, shoulder_y),
            (center_x + 80, shoulder_y + 40),
            (center_x + 75, shoulder_y + 120),
            (center_x + 50, shoulder_y + 100)
        ], fill=color, outline=self.darken_color(color), width=2)
        
        # Knöpfe
        for i in range(5):
            button_y = shoulder_y + 60 + i * 40
            draw.ellipse([center_x - 5, button_y, center_x + 5, button_y + 10], 
                        fill=self.darken_color(color))
    
    def draw_realistic_shoes(self, draw: ImageDraw.Draw, center_x: int, color: Tuple[int, int, int], shoe_type: str):
        """Zeichnet realistische Schuhe"""
        ankle_y = 750
        
        if shoe_type == "Stiefeletten":
            # Stiefeletten (höher)
            boot_height = 80
            draw.polygon([
                (center_x - 35, ankle_y - boot_height),
                (center_x - 15, ankle_y - boot_height),
                (center_x - 10, ankle_y),
                (center_x - 45, ankle_y),
                (center_x - 50, ankle_y - 20)
            ], fill=color, outline=self.darken_color(color), width=2)
            
            draw.polygon([
                (center_x + 15, ankle_y - boot_height),
                (center_x + 35, ankle_y - boot_height),
                (center_x + 50, ankle_y - 20),
                (center_x + 45, ankle_y),
                (center_x + 10, ankle_y)
            ], fill=color, outline=self.darken_color(color), width=2)
        else:
            # Normale Schuhe
            draw.ellipse([center_x - 35, ankle_y - 15, center_x - 5, ankle_y + 15], 
                        fill=color, outline=self.darken_color(color), width=2)
            draw.ellipse([center_x + 5, ankle_y - 15, center_x + 35, ankle_y + 15], 
                        fill=color, outline=self.darken_color(color), width=2)
    
    def add_clothing_details(self, img: Image.Image) -> Image.Image:
        """Fügt Details und Schatten zu Kleidung hinzu"""
        # Weiche Schatten
        shadow_img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Erstelle Schatten-Overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(overlay)
        
        # Subtle Schatten unter Kleidung
        width, height = img.size
        center_x = width // 2
        
        # Schatten unter Oberteil
        shadow_draw.ellipse([center_x - 60, 320, center_x + 60, 350], 
                           fill=(0, 0, 0, 30))
        
        # Composite
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        return img
    
    def add_professional_lighting(self, img: Image.Image) -> Image.Image:
        """Fügt professionelle Beleuchtung hinzu"""
        # Erstelle Beleuchtungs-Overlay
        lighting = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(lighting)
        
        width, height = img.size
        center_x, center_y = width // 2, height // 3
        
        # Hauptlicht (von oben links)
        for radius in range(200, 0, -5):
            alpha = int(20 * (200 - radius) / 200)
            draw.ellipse([center_x - 100 - radius//2, center_y - 100 - radius//2,
                         center_x - 100 + radius//2, center_y - 100 + radius//2], 
                        fill=(255, 255, 255, alpha))
        
        # Fülllicht (von rechts)
        for radius in range(150, 0, -8):
            alpha = int(15 * (150 - radius) / 150)
            draw.ellipse([center_x + 80 - radius//2, center_y - radius//2,
                         center_x + 80 + radius//2, center_y + radius//2], 
                        fill=(255, 240, 200, alpha))
        
        # Composite Beleuchtung
        img = Image.alpha_composite(img.convert('RGBA'), lighting)
        
        return img
    
    def get_clothing_colors_from_style(self, style_prompt: str) -> List[Tuple[int, int, int]]:
        """Realistische Farben basierend auf Fashion-Trends"""
        style_lower = style_prompt.lower()
        
        if "modern" in style_lower or "minimalistisch" in style_lower:
            return [(45, 45, 48), (240, 240, 245), (128, 128, 130)]  # Schwarz, Weiß, Grau
        elif "klassisch" in style_lower:
            return [(25, 25, 112), (255, 255, 255), (139, 69, 19)]  # Navy, Weiß, Braun
        elif "vintage" in style_lower:
            return [(139, 69, 19), (205, 133, 63), (128, 0, 0)]  # Braun, Peru, Maroon
        elif "sportlich" in style_lower:
            return [(0, 0, 0), (255, 69, 0), (255, 255, 255)]  # Schwarz, Rot, Weiß
        elif "extravagant" in style_lower:
            return [(148, 0, 211), (255, 20, 147), (255, 215, 0)]  # Violett, Pink, Gold
        elif "natürlich" in style_lower:
            return [(34, 139, 34), (139, 90, 43), (240, 230, 140)]  # Grün, Braun, Khaki
        elif "monochrom" in style_lower:
            return [(0, 0, 0), (128, 128, 128), (255, 255, 255)]  # Schwarz, Grau, Weiß
        elif "pastell" in style_lower:
            return [(255, 182, 193), (176, 224, 230), (221, 160, 221)]  # Rosa, Hellblau, Plum
        else:
            # Fashion-Default (elegante Erdtöne)
            return [(47, 79, 79), (205, 192, 176), (139, 69, 19)]
    
    def darken_color(self, color: Tuple[int, int, int], factor: float = 0.6) -> Tuple[int, int, int]:
        """Dunkelt eine Farbe ab für Outlines"""
        return tuple(max(0, int(c * factor)) for c in color[:3])
    
    def cleanup_memory(self):
        """Memory Cleanup"""
        if hasattr(self, 'pipe'):
            del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.current_model = None

def create_professional_fashion_prompt(selected_items: List[Dict], style_prompt: str) -> str:
    """Erstellt professionelle Fashion-Photography Prompts"""
    categories = [item['category'] for item in selected_items]
    category_text = ", ".join(set(categories))
    
    # Basis-Prompt für professionelle Fashion-Fotografie
    base_prompt = f"professional fashion photography, elegant female model wearing {category_text}"
    
    # Style-spezifische Ergänzungen
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
    
    # Zusammenfügen
    full_prompt = f"{base_prompt}, {style_addition}, studio lighting, high fashion photography, professional model pose, detailed fabric textures, photorealistic, 8k quality, magazine cover quality"
    
    return full_prompt

def create_professional_negative_prompt() -> str:
    """Erstellt negative Prompts für bessere Qualität"""
    return ("low quality, blurry, distorted, deformed, cartoon, anime, illustration, "
           "amateur photography, bad lighting, overexposed, underexposed, "
           "pixelated, artifacts, bad anatomy, weird proportions, "
           "multiple people, crowd, text, watermark, signature")

# Hauptgenerierungsfunktion (aktualisiert)
def generate_fashion_design_realistic(selected_items: List[Dict], style_prompt: str, 
                                   generator: RealisticFashionGenerator) -> Optional[np.ndarray]:
    """Generiert realistische Fashion-Fotografien"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Schritt 1: Professioneller Prompt
        status_text.text("🎨 Erstelle professionellen Fashion-Prompt...")
        progress_bar.progress(0.2)
        
        prompt = create_professional_fashion_prompt(selected_items, style_prompt)
        negative_prompt = create_professional_negative_prompt()
        
        st.write(f"**Generierungs-Prompt:** {prompt[:100]}...")
        
        # Schritt 2: Versuche SDXL-Turbo
        status_text.text("🖼️ Generiere mit SDXL-Turbo...")
        progress_bar.progress(0.4)
        
        generated_image = generator.generate_with_sdxl_turbo(prompt, negative_prompt)
        
        if generated_image is None:
            # Schritt 3: Versuche Hugging Face API
            status_text.text("🌐 Versuche Hugging Face API...")
            progress_bar.progress(0.6)
            generated_image = generator.generate_with_huggingface_api(prompt)
        
        if generated_image is None:
            # Schritt 4: Professioneller Fallback
            status_text.text("🎨 Erstelle professionelle Fashion-Illustration...")
            progress_bar.progress(0.8)
            generated_image = generator.create_professional_fashion_image(selected_items, style_prompt)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Professionelles Fashion-Design fertig!")
        
        return np.array(generated_image).astype(np.float32) / 255.0
        
    except Exception as e:
        st.error(f"Fehler bei der Generierung: {e}")
        return None
    finally:
        generator.cleanup_memory()

# Utility Functions (erweitert)
def image_to_base64(image_array):
    image = Image.fromarray(image_array.astype(np.uint8))
    image = image.resize((280, 280), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def numpy_to_base64(image_array, size=(512, 768)):
    if len(image_array.shape) == 2:
        image = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
    else:
        image = Image.fromarray((image_array * 255).astype(np.uint8))
    
    if size:
        image = image.resize(size, Image.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Session State und Load Functions (gleich wie vorher)
@st.cache_data(max_entries=1)
def load_fashion_mnist_optimized(sample_size=1000):
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

@st.cache_data(max_entries=1)
def select_random_fashion_items():
    x_all, y_all = load_fashion_mnist_optimized(sample_size=500)
    
    if x_all is None:
        return []
    
    num_items = min(20, len(x_all))
    random_indices = random.sample(range(len(x_all)), num_items)
    
    items = []
    for i, idx in enumerate(random_indices):
        image = x_all[idx]
        label = y_all[idx]
        category = FASHION_CLASSES[label]
        
        brands = ["Chanel", "Dior", "Versace", "Prada", "Gucci", "Armani", "Zara", "H&M"]
        prices = ["45€", "65€", "85€", "120€", "150€", "200€"]
        
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

# Session State initialization
def init_session_state():
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
        st.session_state.generator = RealisticFashionGenerator()
    
    # Limitiere gespeicherte Bilder
    if len(st.session_state.generated_images) > 15:
        st.session_state.generated_images = st.session_state.generated_images[-15:]

# UI Functions
def like_item():
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        current_item = items[st.session_state.current_index]
        st.session_state.liked_items.append(current_item)
        
        if len(st.session_state.all_time_favorites) < 50:
            if not any(fav['original_index'] == current_item['original_index'] 
                      for fav in st.session_state.all_time_favorites):
                st.session_state.all_time_favorites.append(current_item)
        
        st.session_state.current_index += 1

def dislike_item():
    items = st.session_state.fashion_items
    if st.session_state.current_index < len(items):
        st.session_state.current_index += 1

def reset_session():
    st.session_state.current_index = 0
    st.session_state.liked_items = []
    st.session_state.disliked_items = []
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with st.spinner("Lade neue Fashion-Kollektion..."):
        st.session_state.fashion_items = select_random_fashion_items()

# UI Rendering Functions
def render_swipe_tab():
    """Rendert den Swipe-Tab"""
    items = st.session_state.fashion_items
    current_idx = st.session_state.current_index
    total_items = len(items)
    
    if not items:
        st.error("Fehler beim Laden der Fashion-Daten.")
        return
    
    # Verbesserter Progress Bar
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
        
        # Hauptkarte mit verbessertem Design
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
        
        # Verbesserte Buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("👎 Skip", type="secondary", use_container_width=True, help="Nicht interessant"):
                dislike_item()
                st.rerun()
        
        with col3:
            if st.button("❤️ Love it!", type="primary", use_container_width=True, help="Zu Favoriten hinzufügen"):
                like_item()
                st.rerun()
        
        # Keyboard shortcuts info
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; color: #999; font-size: 0.9rem;">
            💡 Tipp: Nutze die Buttons um durch die Fashion-Kollektion zu navigieren
        </div>
        """, unsafe_allow_html=True)

def render_generate_tab():
    """Rendert den überarbeiteten Generate-Tab"""
    st.markdown("## 🎨 Professioneller Fashion-Generator")
    
    # Info-Box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
        <h3 style="margin-top: 0;">✨ Erstelle realistische Fashion-Fotografien</h3>
        <p style="margin-bottom: 0;">Wähle deine Lieblings-Styles und lass AI professionelle Fashion-Bilder generieren, 
        ähnlich wie echte Model-Shootings auf dem Laufsteg.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.all_time_favorites:
        st.warning("💡 Sammle erst Favoriten durch Swipen in der Fashion-Kollektion!")
        return
    
    # Verbesserte Favoriten-Auswahl
    st.markdown("### 👗 Wähle Fashion-Styles (max. 3 für beste Ergebnisse)")
    
    # Erstelle ein Grid für bessere Darstellung
    cols_per_row = 6
    items_to_show = st.session_state.all_time_favorites[:18]  # Zeige mehr Items
    
    for row in range(0, len(items_to_show), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, item in enumerate(items_to_show[row:row + cols_per_row]):
            with cols[idx]:
                is_selected = item in st.session_state.selected_for_generation
                
                # Verbesserter Button-Style
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
                        if len(st.session_state.selected_for_generation) < 5:
                            st.session_state.selected_for_generation.append(item)
                        else:
                            st.warning("⚠️ Maximal 5 Styles für optimale Ergebnisse!")
                    st.rerun()
                
                # Verbesserte Bilddarstellung
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
        
        # Erweiterte Style-Optionen
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
                ["Natürlich", "Monochrom", "Pastell", "Kräftig", "Erdtöne", "Metallic"],
                help="Farbpalette für die Kleidung"
            )
        
        with col3:
            photo_style = st.selectbox(
                "📸 Foto-Stil",
                ["Studio", "Runway", "Street Style", "Editorial", "Commercial"],
                help="Art der Fashion-Fotografie"
            )
        
        # Erweiterte Optionen
        with st.expander("🔧 Erweiterte Einstellungen"):
            col1, col2 = st.columns(2)
            with col1:
                lighting = st.selectbox(
                    "💡 Beleuchtung",
                    ["Professional Studio", "Natural Light", "Dramatic", "Soft"]
                )
            with col2:
                quality_preset = st.selectbox(
                    "⚡ Qualität",
                    ["Standard (schnell)", "Hoch (langsamer)", "Ultra (sehr langsam)"]
                )
        
        style_prompt = f"{style_mood.lower()} {photo_style.lower()} style, {color_scheme.lower()} colors, {lighting.lower()} lighting"
        
        # Verbesserter Generate-Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎨 ✨ Generiere Fashion-Foto!", type="primary", use_container_width=True):
                with st.spinner("Erstelle professionelle Fashion-Fotografie..."):
                    # Generiere mit verbesserter Pipeline
                    generated_image = generate_fashion_design_realistic(
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
                            'lighting': lighting,
                            'items': [item['category'] for item in st.session_state.selected_for_generation],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.session_state.generated_images.append(generation_data)
                        if len(st.session_state.generated_images) > 15:
                            st.session_state.generated_images.pop(0)
                        
                        # Professionelle Darstellung des Ergebnisses
                        st.markdown("---")
                        st.markdown("## 📸 Generiertes Fashion-Foto")
                        
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("""
                            <div class="runway-preview">
                                <div style="position: relative; z-index: 1;">
                            """, unsafe_allow_html=True)
                            
                            st.image(generated_image, 
                                   caption="Professionelles Fashion-Foto (AI-generiert)",
                                   use_column_width=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: white; padding: 20px; border-radius: 15px; 
                                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                                <h4>📋 Foto-Details</h4>
                                <p><strong>Style:</strong> {style_mood} {photo_style}</p>
                                <p><strong>Farben:</strong> {color_scheme}</p>
                                <p><strong>Beleuchtung:</strong> {lighting}</p>
                                <p><strong>Fashion-Items:</strong><br>
                                   {', '.join([item['category'] for item in st.session_state.selected_for_generation])}</p>
                                <p><strong>Generiert:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Download-Button
                            img_buffer = io.BytesIO()
                            img = Image.fromarray((generated_image * 255).astype(np.uint8))
                            img.save(img_buffer, format="PNG", optimize=True, quality=95)
                            img_buffer.seek(0)
                            
                            st.download_button(
                                "💾 Foto herunterladen (PNG)",
                                data=img_buffer,
                                file_name=f"fashion_photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            # Social Share Mockup
                            st.markdown("""
                            <div style="margin-top: 15px; text-align: center;">
                                <small style="color: #666;">📱 Perfekt für Social Media</small>
                            </div>
                            """, unsafe_allow_html=True)

def render_gallery_tab():
    """Zeigt eine professionelle Galerie der generierten Designs"""
    st.markdown("## 🖼️ Fashion-Foto Galerie")
    
    if not st.session_state.generated_images:
        st.markdown("""
        <div style="text-align: center; padding: 60px; color: #666;">
            <h3>📷 Noch keine Fashion-Fotos generiert</h3>
            <p>Gehe zum Generator-Tab und erstelle dein erstes professionelles Fashion-Foto!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"**📸 {len(st.session_state.generated_images)} generierte Fashion-Fotos**")
    
    # Verbesserte Galerie-Darstellung
    for idx, gen_data in enumerate(reversed(st.session_state.generated_images)):
        with st.container():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.image(gen_data['image'], use_column_width=True)
            
            with col2:
                st.markdown(f"""
                <div style="padding: 20px;">
                    <h4>📸 Fashion-Foto #{len(st.session_state.generated_images) - idx}</h4>
                    <p><strong>Style:</strong> {gen_data.get('style', 'N/A')}</p>
                    <p><strong>Farben:</strong> {gen_data.get('colors', 'N/A')}</p>
                    <p><strong>Beleuchtung:</strong> {gen_data.get('lighting', 'N/A')}</p>
                    <p><strong>Items:</strong> {', '.join(gen_data.get('items', []))}</p>
                    <p><strong>Erstellt:</strong> {gen_data.get('timestamp', 'N/A')[:19].replace('T', ' ')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")

def main():
    init_session_state()
    
    # Verbesserter Header
    st.markdown("""
    <div class="main-header">🏃‍♀️ Fashion Swipe Studio</div>
    <div class="sub-header">Professioneller Fashion Generator</div>
    """, unsafe_allow_html=True)
    
    # Navigation mit Icons
    tab1, tab2, tab3 = st.tabs(["🔄 Fashion Swipe", "🎨 AI Generator", "🖼️ Foto-Galerie"])
    
    with tab1:
        render_swipe_tab()
    
    with tab2:
        render_generate_tab()
    
    with tab3:
        render_gallery_tab()
    
    # Verbesserter Sidebar
    with st.sidebar:
        st.markdown("### 📊 Session-Statistiken")
        
        # Schöne Metriken
        col1, col2 = st.columns(2)
        with col1:
            st.metric("❤️ Likes", len(st.session_state.liked_items))
        with col2:
            st.metric("⭐ Favoriten", len(st.session_state.all_time_favorites))
        
        st.metric("📸 Generierte Fotos", len(st.session_state.generated_images))
        
        st.markdown("---")
        
        # Action Buttons
        if st.button("🔄 Neue Session", use_container_width=True):
            reset_session()
            st.rerun()
        
        if st.button("🗑️ Galerie leeren", use_container_width=True):
            st.session_state.generated_images = []
            st.rerun()
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; font-size: 0.8rem;">
            <strong>💡 Über Fashion AI Studio:</strong><br>
            • Realistische Fashion-Fotografie<br>
            • SDXL-Turbo Integration<br>
            • Professionelle Runway-Looks<br>
            • Optimiert für < 5GB RAM
        </div>
        """, unsafe_allow_html=True)
        
        # Optional: System Info
        if st.checkbox("🔧 System Info"):
            try:
                import psutil
                memory = psutil.virtual_memory()
                st.metric("RAM", f"{memory.percent:.1f}%")
                
                if torch.cuda.is_available():
                    st.metric("GPU", "CUDA verfügbar")
                else:
                    st.metric("GPU", "CPU-Modus")
            except ImportError:
                st.caption("psutil nicht verfügbar")

if __name__ == "__main__":
    # Memory Management
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (5 * 1024 * 1024 * 1024, -1))
    except (ImportError, AttributeError):
        pass
    
    main()