# English below 

# 📚 Türkçe Kitap Yorumları Duygu Analizi (Sentiment Analysis)

Bu proje, makine öğrenmesi (ML) ve doğal dil işleme (NLP) tekniklerini kullanarak Türkçe kitap yorumlarının duygu polaritesini (Olumlu/Olumsuz) analiz etmeyi amaçlamaktadır. Proje kapsamında, dengeli iki farklı veri seti (10.402 ve 24.748 yorum) oluşturulmuş ve bu veri setleri üzerinde popüler sınıflandırma algoritmalarının performansları karşılaştırılmıştır.

## 🎯 Temel Amaçlar

* Büyük bir Türkçe yorum veri seti üzerinde temizleme ve ön işleme adımlarını uygulamak.
* Dengeli bir eğitim seti oluşturarak model başarısını artırmak.
* CountVectorizer kullanarak metin verisini makine öğrenmesi algoritmaları için sayısal vektörlere dönüştürmek.
* Farklı sınıflandırma modellerinin performanslarını karşılaştırarak en uygun modeli belirlemek.

## ⚙️ Uygulanan Metodoloji

### Veri Kaynağı ve Ön İşleme

Proje, **bir kitap alışveriş sitesine ait yaklaşık 400.000 yorum** içeren bir veri seti ile başlamıştır. Veri ön işleme ve dengeleme adımları sonrasında temel model eğitimi için **10.402** ve **24.748** yorum içeren dengeli veri setleri kullanılmıştır.

1.  **Veri Yükleme ve Temizleme:** Başlangıç verisi olan `tb1.csv`'deki yinelenen (duplicate) 1.433 adet yorum silinmiştir.
2.  **Dengeleme Stratejisi (İkili Sınıflandırma):**
    * **Dataset 1 (10.402 Yorum):** 1 Puan (Olumsuz $\rightarrow 0$) ve 5 Puan (Olumlu $\rightarrow 1$) yorumlar eşitlenmiştir.
    * **Dataset 2 (24.748 Yorum):** 1 ve 2 Puanlar tek bir Olumsuz $\rightarrow 0$ sınıfında toplanmış, 5 Puanlar ise Olumlu $\rightarrow 1$ sınıfını oluşturmak için eşitlenmiştir.
3.  **Metin İşleme:** Türkçe durak kelimeler (`stopwords`) yorum metinlerinden temizlenmiştir.
4.  **Vektörleştirme:** `CountVectorizer` kullanılarak metinler sayısal özellik matrisine dönüştürülmüştür.

### Makine Öğrenmesi Modelleri

Her iki veri seti (%67 Eğitim / %33 Test) üzerinde aşağıdaki algoritmalar eğitilmiş ve kaydedilmiştir (`.model` ve `vectorizer.pickle` dosyaları):

1.  Lojistik Regresyon (`LogisticRegression`)
2.  K-En Yakın Komşu (`KNeighborsClassifier`)
3.  Karar Ağacı (`DecisionTreeClassifier`)
4.  Rastgele Orman (`RandomForestClassifier`)
5.  Destek Vektör Makineleri (`SVC`)
6.  Multinomial Naive Bayes (`MultinomialNB`)

## 📊 Model Başarılarının Karşılaştırılması (Accuracy Score)

| Model | Dataset 1 (10.402 Yorum) Başarısı | Dataset 2 (24.748 Yorum) Başarısı |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | **0.8695** | (Çalışmaya eklenmemiş) |
| **Lojistik Regresyon** | 0.8567 | **0.8603** |
| **Destek Vektör Makineleri (SVM)** | 0.8485 | 0.8445 |
| **Random Forest** | 0.8375 | 0.8385 |
| **Decision Tree** | 0.7856 | 0.7828 |
| **K-Nearest Neighbors (KNN)** | 0.7098 (k=5) | 0.7191 (k=15) |

**Sonuç:** `Multinomial Naive Bayes` ve `Lojistik Regresyon` algoritmaları, her iki veri seti üzerinde de **%86'nın üzerinde** doğruluk oranı ile en başarılı performansı göstermiştir.

## 🙏 Teşekkürler

Bu veri setinin sağlanmasındaki katkılarından dolayı **https://github.com/malibayram** adresine teşekkürler.

***

## 🇬🇧 English Version

# 📚 Turkish Book Review Sentiment Analysis

This project aims to analyze the sentiment polarity (Positive/Negative) of Turkish book reviews using Machine Learning (ML) and Natural Language Processing (NLP) techniques. Two different balanced datasets (10,402 and 24,748 reviews) were created and the performances of popular classification algorithms were compared on these datasets.

## 🎯 Core Objectives

* To apply cleaning and preprocessing steps on a large Turkish review dataset.
* To create a balanced training set to improve model accuracy.
* To convert text data into numerical vectors for ML algorithms using CountVectorizer.
* To compare the performances of different classification models to determine the most suitable model.

## ⚙️ Methodology Applied

### Data Source and Preprocessing

The project started with a dataset containing **approximately 400,000 reviews from a book shopping website**. After data preprocessing and balancing, balanced datasets containing **10,402** and **24,748** reviews were used for core model training.

1.  **Data Cleaning:** 1,433 duplicate reviews from the initial `tb1.csv` file were dropped.
2.  **Balancing Strategy (Binary Classification):**
    * **Dataset 1 (10,402 Reviews):** 1-star (Negative $\rightarrow 0$) and 5-star (Positive $\rightarrow 1$) reviews were equalized.
    * **Dataset 2 (24,748 Reviews):** 1 and 2-star reviews (total 12,374) were grouped into the Negative $\rightarrow 0$ class, and 5-star reviews (12,374) were equalized to form the Positive $\rightarrow 1$ class.
3.  **Text Processing:** Turkish stopwords were removed from the review texts.
4.  **Vectorization:** Text data was converted into a numerical feature matrix using `CountVectorizer`.

### Machine Learning Models

The following classification algorithms were trained and saved (`.model` and `vectorizer.pickle` files) on both datasets (67% Train / 33% Test):

1.  Logistic Regression
2.  K-Nearest Neighbors (KNN)
3.  Decision Tree
4.  Random Forest
5.  Support Vector Machine (SVM)
6.  Multinomial Naive Bayes (MNB)

## 📊 Comparison of Model Accuracies (Accuracy Score)

| Model | Dataset 1 (10,402 Reviews) Accuracy | Dataset 2 (24,748 Reviews) Accuracy |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | **0.8695** | (Not included in analysis) |
| **Logistic Regression** | 0.8567 | **0.8603** |
| **Support Vector Machine (SVM)** | 0.8485 | 0.8445 |
| **Random Forest** | 0.8375 | 0.8385 |
| **Decision Tree** | 0.7856 | 0.7828 |
| **K-Nearest Neighbors (KNN)** | 0.7098 (k=5) | 0.7191 (k=15) |

**Conclusion:** `Multinomial Naive Bayes` and `Logistic Regression` algorithms showed the best performance with an accuracy rate of **over 86%** on both datasets.

## 🚀 How to Run the Project

1.  Clone the repository:
    ```bash
    git clone [YOUR_REPO_ADDRESS]
    ```
2.  Install the necessary libraries:
    ```bash
    pip install pandas numpy scikit-learn nltk seaborn matplotlib
    ```
3.  Open the Jupyter Notebook file:
    ```bash
    jupyter notebook NLP_Kitap-ML.ipynb
    ```
4.  Run the code cells sequentially to see the creation of the datasets, model training, and testing phases.

## 🙏 Acknowledgements

Thanks to **https://github.com/malibayram** for contributing the dataset.
