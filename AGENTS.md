# Proje Görevi: CMPE 346 - Çok Dilli Duygu Analizi (Assignment 02)

Bu belge, CMPE 346 dersi 2. ödevi kapsamında geliştirilecek olan Çok Dilli Duygu Analizi modelinin teknik gereksinimlerini ve kısıtlamalarını içerir.

## 🎯 Temel Hedef
Farklı dillerdeki metinleri "pozitif" veya "negatif" olarak sınıflandırabilen, Transformer tabanlı bir Çok Dilli Dil Modeli (Multilingual LM) ince ayarı (fine-tuning) yapmak.

## 🛠️ Teknik Gereksinimler

### 1. Preprocessing (`preprocessing.py`)
- Çok dilli (multilingual) bir Tokenizer yaklaşımı benimseyen bir ön işleyici sınıfı geliştirilmelidir.
- Veri seti içerisindeki farklı dillerdeki metinlerin tokenize edilmesi ve modele uygun hale getirilmesi bu sınıfın sorumluluğundadır.

### 2. Model Yapısı (`model.py`)
- İkili duygu sınıflandırması (binary sentiment classification) için bir Transformer modeli kullanılmalıdır.
- **KRİTİK:** Mevcut kodda yer alan `LabelEncoder` sınıfı kesinlikle değiştirilmemelidir (**Do not change LabelEncoder class**).

### 3. Eğitim ve Deneyler (`main.py`)
- `train.csv` ve `valid.csv` dosyaları kullanılarak model eğitilmelidir.
- En iyi F1 skorunu elde etmek için farklı hiperparametreler ve deneyler yapılmalıdır.
- Model, Huggingface Hub üzerinde herkese açık (public) bir repository'de yayınlanmalıdır.

### 4. Değerlendirme (`run_test.py`)
- Eğitim sonrası değerlendirme bu script ile yapılacaktır.
- **KRİTİK:** Değerlendirme kodları (evaluation codes) üzerinde hiçbir değişiklik yapılmamalıdır. Modelin bu script ile sorunsuz çalıştığından emin olunmalıdır.

## 📂 Dosya Yapısı
Teslim edilecek `.zip` dosyası aşağıdaki hiyerarşiye sahip olmalıdır:

```text
[student_id_assignment02].zip
├── data/
│   ├── train.csv
│   └── valid.csv
├── main.py
├── model.py
├── preprocessing.py
├── run_test.py
└── README.md
