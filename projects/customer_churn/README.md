# 🏦 Bank Customer Churn Prediction

Bank mijozlarining ketish (churn) ehtimolini bashorat qiluvchi Machine Learning loyihasi. Loyiha to'liq ML siklini qamrab oladi: ma'lumot tahlili (EDA), preprocessing, uchta modelni solishtirish, test to'plami uchun prognoz.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)

---

## 🎯 Biznes muammosi

Yangi mijozni jalb qilish mavjud mijozni ushlab qolishdan bir necha barobar qimmatga tushadi. Agar bank ketish xavfi yuqori mijozlarni **oldindan** aniqlay olsa, ularga maxsus taklif berib, daromadni saqlab qolishi mumkin.

**Vazifa:** mijozning demografik va bank ma'lumotlari asosida uning bankni tark etish ehtimolini (0 dan 1 gacha) bashorat qilish. Bu binary classification masalasi.

## 📊 Ma'lumotlar

| Fayl | Hajm | Tavsif |
|---|---|---|
| `train.csv` | 15 000 × 14 | O'qitish uchun, `Exited` (1 = ketgan, 0 = qolgan) ustuni bilan |
| `test.csv` | 10 000 × 13 | Prognoz qilish uchun, `Exited` ustunisiz |
| `sample_submission.csv` | 10 000 × 2 | Yakuniy natija formati (`id`, `Exited` ehtimoli) |

**Belgilar:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

Ma'lumotda bo'sh qiymatlar va dublikatlar yo'q. Mijozlarning **20%** i ketgan, ya'ni ma'lumot nomutanosib (imbalanced).

##  Asosiy topilmalar (EDA)

![Churn by features](images/churn_by_features.png)

- **Mahsulotlar soni** eng kuchli omil. 2 ta mahsulotli mijozlar atigi 4% ketadi, 1 ta mahsulotlilar 38%, 3 va undan ko'p mahsulotlilar esa 85–100%. Bog'liqlik chiziqli emas.
- **Germaniya:** churn 41%, Fransiya va Ispaniyada 15%.
- **Faol bo'lmagan mijozlar** faollarga qaraganda 2 barobar ko'p ketadi.
- **Yosh:** ketgan mijozlarning median yoshi 45, qolganlarniki 35.
- Kredit kartasi borligi (HasCrCard) va bankdagi yillar (Tenure) churn'ga deyarli ta'sir qilmaydi.

##  Yondashuv

1. **Preprocessing:** identifikator ustunlarini olib tashlash (`id`, `CustomerId`, `Surname`) → One-Hot Encoding (`Geography`, `Gender`) → stratified 80/20 train/validation bo'lish → StandardScaler (faqat train'da `fit`, data leakage'ning oldini olish uchun).
2. **Modellar:** Logistic Regression, Random Forest, XGBoost.
3. **Baholash:** Accuracy, Precision, Recall, F1-score, ROC-AUC va Confusion Matrix. Ma'lumot nomutanosib bo'lgani va natija ehtimol ko'rinishida kutilgani uchun asosiy metrika sifatida **ROC-AUC** tanlandi.

##  Natijalar (Validation, 3 000 mijoz)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost**  | 0.895 | 0.767 | **0.679** | **0.720** | **0.933** |
| Random Forest | 0.897 | 0.796 | 0.647 | 0.714 | 0.932 |
| Logistic Regression | 0.874 | 0.751 | 0.543 | 0.630 | 0.884 |

![Model comparison](images/model_comparison.png)

Daraxtga asoslangan modellar Logistic Regressiondan ancha yaxshi chiqdi, chunki `NumOfProducts` va `Age` churn bilan chiziqli bo'lmagan bog'liqlikka ega.

<p float="left">
  <img src="images/feature_importance.png" width="52%" />
  <img src="images/confusion_matrix.png" width="40%" />
</p>

###  Biznes uchun ma'nosi

- Model eng xavfli deb topgan **20% mijoz** ichida barcha ketuvchilarning **75%** i bor. Ya'ni bank faqat har beshinchi mijoz bilan ishlab, ketayotganlarning to'rtdan uch qismini qamrab oladi.
- Eng xavfli **10% mijozning 91%** i haqiqatan ketgan. Bu guruhga yo'naltirilgan kampaniya deyarli bekor ketmaydi.
- **Maqsadli segment:** 40+ yoshli, faol bo'lmagan, 1 ta mahsulotga ega va Germaniyadagi mijozlar. Ularga ikkinchi mahsulot taklif qilish eng samarali strategiya bo'lishi mumkin, chunki 2 ta mahsulotli mijozlar deyarli ketmaydi.


## 📁 Loyiha tuzilmasi

```
├── bank_churn.ipynb       # To'liq tahlil: EDA → Preprocessing → Modellar → Prognoz
├── app.py                 # Streamlit web ilova
├── submission.csv         # test.csv uchun 10 000 ta prognoz
├── requirements.txt       # Kerakli kutubxonalar
├── data/                  # train.csv, test.csv, sample_submission.csv
├── models/                # churn_model.pkl, scaler.pkl, feature_columns.pkl
└── images/                # README uchun grafiklar
```

##  Ishga tushirish

```bash
git clone https://github.com/<username>/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt

jupyter notebook bank_churn.ipynb   # Notebook'ni ochish
```

##  Keyingi qadamlar

- Preprocessing va modelni bitta `sklearn.Pipeline`'ga birlashtirish
- Stratified K-Fold Cross-Validation bilan natijalarni mustahkamlash
- `GridSearchCV` yoki Optuna bilan hyperparametr tuning
- Biznes xarajatlariga qarab qaror chegarasini (threshold) optimallashtirish
- SHAP yordamida har bir prognozni tushuntirish
- Streamlit Community Cloud'ga deploy qilish

##  Texnologiyalar

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Joblib · Google Colab

##  Muallif

**[Izzatullo Jakbarov]**
[LinkedIn](https://www.linkedin.com/in/izzat-jakbarov-12a461270/) · [Telegram](https://t.me/uzz19_11) · [Email](izzatjakbarov7@gmail.com)
