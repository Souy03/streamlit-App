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

class SimpleFashionGenerator:
    """Einfacher Fashion Generator ohne schwere Dependencies"""
    
    def __init__(self):
        pass
    
    def create_fashion_design(self, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Erstellt Fashion-Design"""
        
        # Canvas erstellen
        img = Image.new('RGB', (512, 768), color=(245, 245, 250))
        
        # Hintergrund
        img = self.create_background(img)
        
        # Model
        model_img = self.create_model(img.size)
        
        # Styling
        model_img = self.apply_styling(model_img, selected_items, style_prompt)
        
        # Combine
        final_img = Image.alpha_composite(img.convert('RGBA'), model_img.convert('RGBA'))
        
        return final_img.convert('RGB')
    
    def create_background(self, base_img: Image.Image) -> Image.Image:
        """Erstellt Hintergrund"""
        draw = ImageDraw.Draw(base_img)
        width, height = base_img.size
        
        # Gradient
        for y in range(height):
            shade = int(240 + (y / height) * 15)
            draw.rectangle([0, y, width, y+1], fill=(shade, shade, shade+5))
        
        return base_img
    
    def create_model(self, size: Tuple[int, int]) -> Image.Image:
        """Erstellt Model"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center_x = width // 2
        skin_color = (245, 220, 177)
        
        # Einfache Model-Form
        # Kopf
        draw.ellipse([center_x - 25, 60, center_x + 25, 110], fill=skin_color)
        
        # Körper
        draw.rectangle([center_x - 30, 110, center_x + 30, 400], fill=skin_color)
        
        # Arme
        draw.rectangle([center_x - 60, 120, center_x - 30, 300], fill=skin_color)
        draw.rectangle([center_x + 30, 120, center_x + 60, 300], fill=skin_color)
        
        # Beine
        draw.rectangle([center_x - 25, 400, center_x - 5, 700], fill=skin_color)
        draw.rectangle([center_x + 5, 400, center_x + 25, 700], fill=skin_color)
        
        return img
    
    def apply_styling(self, model_img: Image.Image, selected_items: List[Dict], style_prompt: str) -> Image.Image:
        """Wendet Styling an"""
        draw = ImageDraw.Draw(model_img)
        width, height = model_img.size
        center_x = width // 2
        
        # Farben basierend auf Style
        colors = self.get_colors(style_prompt)
        
        categories = [item['category'] for item in selected_items]
        
        for i, category in enumerate(set(categories)):
            color = colors[i % len(colors)]
            
            if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
                # Oberteil
                draw.rectangle([center_x - 28, 120, center_x + 28, 250], fill=color)
            elif category == "Kleid":
                # Kleid
                draw.polygon([
                    (center_x - 28, 120),
                    (center_x + 28, 120),
                    (center_x + 35, 400),
                    (center_x - 35, 400)
                ], fill=color)
            elif category == "Hose":
                # Hose
                draw.rectangle([center_x - 23, 250, center_x - 7, 500], fill=color)
                draw.rectangle([center_x + 7, 250, center_x + 23, 500], fill=color)
            elif category == "Mantel":
                # Mantel
                draw.polygon([
                    (center_x - 35, 115),
                    (center_x + 35, 115),
                    (center_x + 40, 450),
                    (center_x - 40, 450)
                ], fill=color)
        
        return model_img
    
    def get_colors(self, style_prompt: str) -> List[Tuple[int, int, int]]:
        """Gibt Farben zurück"""
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

def create_sample_fashion_items():
    """Erstellt Sample Fashion Items ohne TensorFlow"""
    items = []
    brands = ["Chanel", "Dior", "Versace", "Prada", "Gucci", "Armani", "Zara", "H&M"]
    prices = ["45€", "65€", "85€", "120€", "150€", "200€"]
    
    for i in range(15):
        # Erstelle ein einfaches Sample-Bild
        category_idx = i % len(FASHION_CLASSES)
        category = FASHION_CLASSES[category_idx]
        
        # Einfaches Bild erstellen
        img = Image.new('L', (28, 28), color=128)
        draw = ImageDraw.Draw(img)
        
        # Einfache Form je nach Kategorie
        if category in ["T-Shirt/Top", "Hemd", "Pullover"]:
            draw.rectangle([5, 8, 23, 20], fill=200)
        elif category == "Kleid":
            draw.polygon([(8, 8), (20, 8), (22, 24), (6, 24)], fill=200)
        elif category == "Hose":
            draw.rectangle([8, 12, 12, 26], fill=200)
            draw.rectangle([16, 12, 20, 26], fill=200)
        else:
            draw.ellipse([6, 6, 22, 22], fill=200)
        
        item = {
            "id": i + 1,
            "name": f"{category} Collection #{i}",
            "brand": random.choice(brands),
            "price": random.choice(prices),
            "category": category,
            "description": f"Premium {category.lower()} aus der aktuellen Kollektion",
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
    image.save(buffer, format="PNG")
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
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

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
        st.session_state.generator = SimpleFashionGenerator()

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
    st.markdown("## 🎨 Fashion-Generator")
    
    st.markdown("""
    <div class="generation-alert">
        <h3>🎨 Lokaler Fashion-Generator</h3>
        <p>Erstellt stylische Fashion-Designs basierend auf deinen Favoriten!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.all_time_favorites:
        st.warning("💡 Sammle erst Favoriten durch Swipen!")
        return
    
    # Favoriten-Auswahl
    st.markdown("### 👗 Wähle Fashion-Styles")
    
    cols = st.columns(5)
    for idx, item in enumerate(st.session_state.all_time_favorites[:10]):
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
                        st.warning("Maximal 3 Styles!")
                st.rerun()
            
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="{item['image_data']}" 
                     style="width: 100%; border-radius: 8px;
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
            style_mood = st.selectbox("🎭 Style", ["Modern", "Klassisch", "Vintage", "Sportlich"])
        
        with col2:
            color_scheme = st.selectbox("🎨 Farben", ["Natürlich", "Monochrom", "Pastell", "Kräftig"])
        
        # Generate-Button
        st.markdown("---")
        if st.button("🎨 ✨ Generiere Fashion-Design!", type="primary", use_container_width=True):
            with st.spinner("Erstelle Fashion-Design..."):
                style_prompt = f"{style_mood.lower()} style, {color_scheme.lower()} colors"
                
                generated_image = st.session_state.generator.create_fashion_design(
                    st.session_state.selected_for_generation,
                    style_prompt
                )
                
                if generated_image is not None:
                    # Speichere Generierung
                    generation_data = {
                        'image': numpy_to_base64(np.array(generated_image) / 255.0),
                        'style': style_mood,
                        'colors': color_scheme,
                        'items': [item['category'] for item in st.session_state.selected_for_generation],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.generated_images.append(generation_data)
                    
                    # Darstellung
                    st.markdown("---")
                    st.markdown("## 📸 Generiertes Fashion-Design")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.image(generated_image, caption="Fashion-Design", use_column_width=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: white; padding: 20px; border-radius: 15px;">
                            <h4>📋 Design-Details</h4>
                            <p><strong>Style:</strong> {style_mood}</p>
                            <p><strong>Farben:</strong> {color_scheme}</p>
                            <p><strong>Items:</strong> {', '.join([item['category'] for item in st.session_state.selected_for_generation])}</p>
                        </div>
                        """, unsafe_allow_html=True)

def render_gallery_tab():
    """Zeigt Galerie"""
    st.markdown("## 🖼️ Design-Galerie")
    
    if not st.session_state.generated_images:
        st.info("Noch keine Designs generiert!")
        return
    
    for idx, gen_data in enumerate(reversed(st.session_state.generated_images)):
        with st.container():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.image(gen_data['image'])
            
            with col2:
                st.markdown(f"""
                **Design #{len(st.session_state.generated_images) - idx}**
                - Style: {gen_data.get('style', 'N/A')}
                - Farben: {gen_data.get('colors', 'N/A')}
                - Items: {', '.join(gen_data.get('items', []))}
                """)
            
            st.markdown("---")

def main():
    """Hauptfunktion"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">🏃‍♀️ Fashion Swipe Studio</div>
    <div class="sub-header">Minimal Working Version</div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔄 Fashion Swipe", "🎨 Generator", "🖼️ Galerie"])
    
    with tab1:
        render_swipe_tab()
    
    with tab2:
        render_generate_tab()
    
    with tab3:
        render_gallery_tab()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Statistiken")
        
        st.metric("❤️ Likes", len(st.session_state.liked_items))
        st.metric("⭐ Favoriten", len(st.session_state.all_time_favorites))
        st.metric("📸 Designs", len(st.session_state.generated_images))
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            reset_session()
            st.rerun()

if __name__ == "__main__":
    main()
