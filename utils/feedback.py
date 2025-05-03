import pandas as pd
import os
from datetime import datetime
from config import Config

def save_feedback(image_path, predicted_class, confidence, user_feedback, correct_class=None):
    """Kullanıcı feedback'ini kaydeder"""
    # Feedback veri yapısı
    feedback_data = {
        'image_path': image_path,
        'timestamp': datetime.now().isoformat(),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'user_feedback': user_feedback,
        'correct_class': correct_class if user_feedback == 'incorrect' else predicted_class
    }
    
    # Feedback dizinini oluştur
    os.makedirs(Config.FEEDBACK_DIR, exist_ok=True)
    
    # CSV dosyasına ekle
    if os.path.exists(Config.FEEDBACK_FILE):
        df = pd.read_csv(Config.FEEDBACK_FILE)
    else:
        df = pd.DataFrame(columns=feedback_data.keys())
    
    df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
    df.to_csv(Config.FEEDBACK_FILE, index=False)

def analyze_feedback():
    """Feedback verilerini analiz eder"""
    if not os.path.exists(Config.FEEDBACK_FILE):
        return None
    
    df = pd.read_csv(Config.FEEDBACK_FILE)
    
    # Temel analizler
    analysis = {
        'total_feedbacks': len(df),
        'correct_rate': len(df[df['user_feedback'] == 'correct']) / len(df),
        'common_errors': df[df['user_feedback'] == 'incorrect']['correct_class'].value_counts().to_dict(),
        'low_confidence': df[df['confidence'] < 70].sort_values('confidence').head(10)
    }
    
    return analysis