import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow loglarını azalt
import warnings
warnings.filterwarnings("ignore")  # İstenmeyen uyarıları gizle

import streamlit as st

# SET_PAGE_CONFIG MUTLAKA EN ÜSTTE VE İLK STREAMLIT KOMUTU OLMALI
st.set_page_config(
    page_title="Alzheimer MRI Sınıflandırıcı",
    page_icon="🧠",
    layout="wide"
)

# Diğer importlar SET_PAGE_CONFIG'tan sonra gelmeli
from PIL import Image
import numpy as np
import tensorflow as tf
from config import Config
import pandas as pd

# Streamlit context uyarılarını önle
try:
    import streamlit.runtime.scriptrunner as scriptrunner
    scriptrunner._thread_local = scriptrunner._ThreadLocal()
except:
    pass

# Yol kontrollerini yap
Config.check_paths()

# Modeli yükle
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(Config.MODEL_PATH)
        class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        return model, class_names
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        st.stop()

model, class_names = load_model()

# Streamlit arayüzü
st.title('🧠 Alzheimer MRI Sınıflandırıcı')
st.write("Bu uygulama, MRI görüntülerine göre Alzheimer evrelerini sınıflandırır.")

# Görsel yükleme
uploaded_file = st.file_uploader("Bir MRI görüntüsü yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli göster
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Görüntü', use_column_width=True)
    
    # Tahmin yap
    if st.button('Sınıflandır'):
        with st.spinner('Analiz ediliyor...'):
            try:
                # Görseli ön işleme
                img = img.resize(Config.IMG_SIZE)
                img_array = np.array(img) / 255.0
                
                # RGB kontrolü
                if len(img_array.shape) == 2:  # Grayscale ise
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA ise
                    img_array = img_array[:,:,:3]
                
                img_array = np.expand_dims(img_array, axis=0)
                
                # Tahmin
                predictions = model.predict(img_array)[0]
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                # Sonuçları göster
                st.success(f"**Tahmin:** {predicted_class} (%{confidence:.1f} güven)")
                
                # Tüm sınıflar için olasılıklar
                st.subheader("Tüm Sınıf Olasılıkları")
                prob_df = pd.DataFrame({
                    'Sınıf': class_names,
                    'Olasılık': predictions
                }).sort_values('Olasılık', ascending=False)
                st.bar_chart(prob_df.set_index('Sınıf'))
                
            except Exception as e:
                st.error(f"Tahmin yapılırken hata oluştu: {str(e)}")

# Yan bilgi çubuğu
st.sidebar.header("Dataset Bilgisi")
st.sidebar.info("""
Bu uygulama [Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) kullanılarak eğitilmiştir.

**Sınıflar:**
- MildDemented
- ModerateDemented
- NonDemented
- VeryMildDemented
""")
