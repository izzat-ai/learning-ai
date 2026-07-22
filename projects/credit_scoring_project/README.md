# Bank Credit Risk Assessment & Income Prediction

> **Loyiha maqsadi:** Demografik va moliyaviy ko'rsatkichlar asosida mijozlarning daromad darajasini (`<=50K` yoki `>50K`) bashorat qilish, kredit ajratishdagi xavflarni baholash hamda bankning moliyaviy xavfsizligini ta'minlash.

---

## Loyiha Haqida

Ushbu loyihada ijtimoiy-iqtisodiy omillar *(yosh, ma'lumot darajasi, kasb, haftalik ish soati, kapital tushumi va hokazo)* yordamida mijozning to'lov qobiliyatini aniqlovchi **Machine Learning** va **Ensemble modellari** tizimi ishlab chiqildi.

* **Dataset:** Census Income Dataset (Adult Dataset)
* **Masala turi:** Binar Klassifikatsiya (Binary Classification)
* **Biznes Strategiyasi:** Bank risk-menejmenti talablariga mos ravishda xato kredit berish ehtimolini kamaytirish va umumiy model aniqligini optimallashtirish.

---

## Texnologiyalar va Kutubxonalar

* **Data Analysis & Vizualizatsiya:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Data Preprocessing & Pipeline:** `Scikit-Learn` (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`)
* **Machine Learning Algoritmlari:** `LogisticRegression`, `RandomForestClassifier`, `XGBoostClassifier`

---

## Loyiha Bosqichlari
1. **EDA & Data Cleaning:** 
   Bo'sh (NaN) qiymatlarni aniqlandi, belgilarni tahlil qilindi hamda disbalans tarqalishi o'rganildi.
2. **Data Preprocessing Pipeline (Data Leakage'siz):**
   * **Sonli ustunlar:**  `StandardScaler()`
   * **Kategoriyali ustunlar:** `SimpleImputer(strategy='constant', fill_value='Unknown')` + `OneHotEncoder(handle_unknown='ignore')`
3. **Model Training & Evaluation:**
   Ma'lumotlarni proporsional ravishda bo'lindi (`stratify=y`), modellar o'qitildi hamda `Confusion Matrix` va `Classification Report` orqali solishtirildi.

---

## Modellarning Batal Taqqoslamasi

> **Biznes konteksti:** Bank uchun eng xavfli holat — kam daromadli mijozga adashib kredit ajratishdir . Shuning uchun modelni tanlashda **Precision** va umumiy **Accuracy** asosiy mezon hisoblandi.

---

## Asosiy Biznes Xulosalari 

* ** Xavfsizlik bo'yicha — Random Forest Classifier:**
  * `>50K` guruhida **Precision = 81.8%** ko'rsatkichini berdi.
  * U bor-yo'g'i **295 ta** kam daromadli mijozga adashib kredit berdi. Bank kapitalini xavf-xatardan saqlash uchun eng xavfsiz model.

* ** Daromad va Aniqlik bo'yicha — XGBoost:**
  * Umumiy **Accuracy (87.46%)** va **F1-Score (0.71)** bo'yicha yetakchi bo'ldi .
  * Random Forest'ga qaraganda **166 ta ko'proq** ishonchli mijozni aniqladi (**Recall = 63.7%**), bankning kredit ajratish va foizli daromad olish imkoniyatlarini oshirdi.
