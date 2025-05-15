# meyveSiniflandirma
Yapay Zeka için-2025 Repository.

📌 Proje Hakkında
Bu proje, Alzheimer Augmented MRI Dataset kullanarak Alzheimer's hastalığının 4 farklı evresini tespit etmeyi hedefleyen bir uygulamadır. Uygulama özellikleri, teknolojileri ve dosya yapısı belirtildiği gibi olmak zorundadır!

✨ Özellikler
🖼️ Kullanıcı dostu görsel yükleme arayüzü

🤖 EfficientNetB0 tabanlı derin öğrenme modeli

📊 Hangi Sınıfa Yüzde Kaç Yaklaşıldı

📈 Model performans metrikleri

4 Farklı Alzheimer Evre Tespit Sistemi

🛠️ Teknoloji Yığını

Python (3.8+)

TensorFlow (2.12+)

Streamlit (1.25+)

Pandas (1.5+)

NumPy (1.24+)

Pillow (9.5+)

🚀 Kurulum
Ön Koşullar
Python 3.8 veya üzeri

pip paket yöneticisi

Alzheimer (https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) Dataset indirme işlemi.

İndirilen dosyaların ayrı klasör içerisinde "data" klasörü oluşturularak atılması.

train.py çalıştırılması.

app.py çalıştırılması.

Dosya yapısı şöyle olmalıdır :

alzheimer_file
  data
    Training
      Augmented Files
    Test
      Non Augmented
  logs
  models
  utils
    data_loader.py
    model_utils.py
  app.py
  config.py
  train.py
