import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from utils.data_loader import get_class_names
from utils.feedback import save_feedback
from config import Config
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Sayfa ayarları
    st.set_page_config(
        layout="wide", 
        page_title="Fruits 360 Sınıflandırıcı",
        page_icon="🍎"
    )

    # Model ve sınıf isimlerini yükle
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model(Config.MODEL_PATH)
        class_names = get_class_names()
        return model, class_names

    model, class_names = load_model()

    # Uygulama arayüzü
    st.title('🍎 Fruits 360 Sınıflandırıcı 🍌')
    st.write(f"Bu uygulama, {len(class_names)} farklı meyve ve sebze türünü sınıflandırabilir.")

    # Ana sütunlar
    col1, col2 = st.columns([1, 2])

    with col1:
        # Görsel yükleme alanı
        uploaded_file = st.file_uploader(
            "Bir meyve/sebze görseli yükleyin...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Geçici dosyayı kaydet
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Görseli göster
            img = Image.open(uploaded_file)
            st.image(img, caption='Yüklenen Görüntü', use_column_width=True)
            
            # Tahmin butonu
            if st.button('Sınıflandır', use_container_width=True):
                with st.spinner('Model çalışıyor...'):
                    # Görseli ön işleme
                    img = img.resize(Config.IMG_SIZE)
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Tahmin yap
                    predictions = model.predict(img_array)[0]
                    top5_idx = np.argsort(predictions)[-5:][::-1]
                    top5_classes = [class_names[i] for i in top5_idx]
                    top5_probs = predictions[top5_idx]
                    
                    # Sonuçları göster
                    st.success(f"**En Yüksek Tahmin:** {top5_classes[0].capitalize()} (%{top5_probs[0]*100:.1f})")
                    
                    # Top 5 tahmin
                    st.subheader("En Olası 5 Sınıf")
                    prob_df = pd.DataFrame({
                        'Sınıf': [cls.capitalize() for cls in top5_classes],
                        'Olasılık': top5_probs
                    })
                    st.bar_chart(prob_df.set_index('Sınıf'))
                    
                    # Feedback mekanizması
                    st.subheader("Tahmini Değerlendirin")
                    
                    # Doğru tahmin butonu
                    if st.button("👍 Doğru Tahmin", key="correct"):
                        save_feedback(
                            image_path=temp_path,
                            predicted_class=top5_classes[0],
                            confidence=top5_probs[0]*100,
                            user_feedback='correct'
                        )
                        st.success("Teşekkürler! Geri bildiriminiz kaydedildi.")
                    
                    # Yanlış tahmin formu
                    with st.form("incorrect_feedback"):
                        correct_class = st.selectbox(
                            "Doğru sınıfı seçin",
                            options=sorted([cls.capitalize() for cls in class_names])
                        )
                        
                        submitted = st.form_submit_button("👎 Yanlış Tahmin")
                        if submitted:
                            save_feedback(
                                image_path=temp_path,
                                predicted_class=top5_classes[0],
                                confidence=top5_probs[0]*100,
                                user_feedback='incorrect',
                                correct_class=correct_class.lower()
                            )
                            st.success("Teşekkürler! Geri bildiriminiz kaydedildi. Model iyileştirmelerinde kullanılacak.")

    with col2:
        # Model bilgileri
        st.subheader("Model Performansı")
        
        # Confusion matrix
        st.image('confusion_matrix.png', use_column_width=True)
        
        # Eğitim grafikleri
        st.image('training_history.png', use_column_width=True)
        
        # Sınıf sayısı ve örnekler
        st.subheader(f"Desteklenen {len(class_names)} Meyve/Sebze")
        
        # Kategorilere göre gruplandırma (örneğin tüm elma çeşitleri)
        fruit_categories = {}
        for name in class_names:
            main_category = name.split(' ')[0]  # İlk kelimeyi ana kategori olarak al
            if main_category not in fruit_categories:
                fruit_categories[main_category] = []
            fruit_categories[main_category].append(name)
        
        # Kategori seçimi
        selected_category = st.selectbox(
            "Kategori Seçin",
            options=sorted(fruit_categories.keys())
        )
        
        # Seçili kategorideki meyveleri göster
        st.write(f"**{selected_category.capitalize()} Kategorisindeki Türler:**")
        cols = st.columns(3)
        for i, fruit in enumerate(sorted(fruit_categories[selected_category])):
            cols[i%3].write(f"- {fruit.capitalize()}")

    # Yan bilgi çubuğu
    st.sidebar.header("Fruits 360 Dataset")
    st.sidebar.info("""
    Bu uygulama [Fruits 360 dataseti](https://www.kaggle.com/datasets/moltean/fruits) kullanılarak eğitilmiştir.

    **Dataset Özellikleri:**
    - 130+ meyve ve sebze türü
    - 70,000+ yüksek kaliteli görüntü
    - Her sınıfta en az 490 örnek
    - 100x100 piksel çözünürlük
    - Arka plan çıkarılmış görseller
    """)

    # Feedback verilerini göster (geliştirici modu)
    if st.sidebar.checkbox("Geliştirici Modu"):
        try:
            feedback_df = pd.read_csv(Config.FEEDBACK_FILE)
            st.sidebar.subheader("Toplanan Geri Bildirimler")
            
            # Feedback analizi
            correct_rate = len(feedback_df[feedback_df['user_feedback'] == 'correct']) / len(feedback_df)
            st.sidebar.metric("Doğru Tahmin Oranı", f"{correct_rate:.1%}")
            
            # En çok hata yapılan sınıflar
            errors = feedback_df[feedback_df['user_feedback'] == 'incorrect']
            if not errors.empty:
                st.sidebar.subheader("En Çok Hata Yapılan Sınıflar")
                error_counts = errors['correct_class'].value_counts().head(10)
                st.sidebar.bar_chart(error_counts)
        except FileNotFoundError:
            st.sidebar.warning("Henüz geri bildirim toplanmadı")

if __name__ == "__main__":
    main()