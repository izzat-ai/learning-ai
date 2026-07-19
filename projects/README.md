#  TMDB Movies Rating Prediction (Machine Learning)

Ushbu loyihada **The Movie Database (TMDB)** datasetidan foydalanib, filmlarning o'rtacha reytingini (`vote_average`) bashorat qiluvchi Regressiya modeli qurildi. Loyiha ma'lumotlarni tozalashdan boshlab, modelni optimallashtirishgacha bo'lgan to'liq Data Science pipeline-ni o'z ichiga oladi.

## Texnologiyalar
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

## Loyiha Bosqichlari va Amaliy Yechimlar

### 1. Data Cleaning & Outlier Detection
* Datasetdagi dublikatlar va bo'sh qiymatlar tozalandi.
* Grafik tahlillar yordamida umuman ovoz berilmagan, sun'iy ravishda `0` reyting olgan 58 ta anomal qator (outliers) aniqlanib, datasetdan olib tashlandi. Yakuniy dataset **1930 ta sifatli qator** holatiga keltirildi.

### 2. Feature Engineering
* `release_date` ustunidan model aniqligini oshirish uchun **yil**, **oy** va **hafta kuni** kabi yangi raqamli xususiyatlar ajratib olindi.

### 3. Categorical Encoding (Tillari muammosi)
* Datasetda 28 tadan ortiq noyob tillar borligi aniqlandi. Ularning ko'pchiligi juda kam (10 tadan kam) qatnashgani uchun **"Top10 + Others"** strategiyasi qo'llanildi. Eng ko'p uchraydigan 10 ta til saqlab qolinib, qolganlari `other` guruhiga birlashtirildi va **One-Hot Encoding** qilindi.

### 4. Exploratory Data Analysis (EDA)
* Korrelyatsiya tahlilida `vote_count` (ovozlar soni) va `vote_average` o'rtasida **0.39 lik ijobiy bog'liqlik** borligi aniqlandi.

### 5. Model Training & Hyperparameter Tuning
* Model sifatida **Random Forest Regressor** tanlandi va ma'lumotlar 80/20 nisbatda Train/Test-ga ajratildi.
* Model aniqligini oshirish va Overfitting-ning oldini olish uchun **RandomizedSearchCV** yordamida giperparametrlar sozlandi (`n_estimators: 150`, `min_samples_leaf: 8`).

## Model Natijalari

* **Baza Modeli:** R^2 Score = `0.3929`, RMSE = `0.7687` ball
* **Sozlangan Model:** R^2 Score = `0.3888`, RMSE = `0.7713` ball
* **Feature Importance:** Model qaror qabul qilishda eng ko'p `vote_count` va `release_year` ustunlariga tayangani aniqlandi.

## Kelajakdagi Rivojlanish 
Model ko'rsatkichini yanada oshirish uchun kelajakda film janrlari (Comedy, Action va h.k.) va film byudjeti kabi qo'shimcha subyektiv xususiyatlarni ham modelga qo'shish rejalashtirildi.
