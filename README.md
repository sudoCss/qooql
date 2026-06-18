<div dir="rtl">

# قووقل Qooql

## لتشغيل المشروع اتبع الخطوات التالية:

### 1. قم بتنزيل المشروع من git

```bash
git clone https://github.com/sudoCss/qooql.git
cd qooql
```

### 2. قم بانشاء وتفعيل بيئة بايثون افتراضية باستخدام التعليمة المناسبة لنظام التشغيل

```bash
python -m venv .venv
source .venv/bin/activate # THIS IS FOR LINUX
.venv/Scripts/activate.bat # THIS IS FOR WINDOWS CMD
.venv/Scripts/Activate.ps1 # THIS IS FOR WINDOWS POWERSHELL
```

### 3. قم بتنزيل المكاتب اللازمة

```bash
pip install numpy nltk fastapi ir_datasets "setuptools<82" joblib scikit-learn rank_bm25 sentence_transformers faiss-cpu tkinter requests fastapi uvicorn pydantic tqdm pandas
```

### 4. قم بتهيئة قاعدة البيانات

```bash
python -m db.setup
```

### 5. قم بتشغيل الخدمات (كل خدمة في تيرمينال/CMD منفصلة)

```bash
python -m uvicorn api.data_loader_api:app --host 0.0.0.0 --port 8001
python -m uvicorn api.representation_api:app --host 0.0.0.0 --port 8002
python -m uvicorn api.search_api:app --host 0.0.0.0 --port 8003
```

ملاحظة هامة جداً: يجب في كل تيرمينال/CMD جديدة تقوم بفتحها في مجلد المشروع أن تقوم بتفعيل البيئة الافتراضية ليتم التعرف على المكاتب والمسارات

```bash
source .venv/bin/activate # THIS IS FOR LINUX
.venv/Scripts/activate.bat # THIS IS FOR WINDOWS CMD
.venv/Scripts/Activate.ps1 # THIS IS FOR WINDOWS POWERSHELL
```

### 6. قم بتشغيل الواجهة

```bash
python -m ui.app
```

### 7. في الواجهة يمكنك تنزيل الـ Datasets المتاحة وبناء تمثيلاتها المطلوبة باستخدام الأزرار

### 8. بعد التنزيل والتدريب(بناء ملفات التمثيلات المطلوبة) يمكنك اجراء عمليات البحث

### 9. قم بتشغيل عملية الاختبار واختر الـ Dataset الجاهزة عندك ليتم حساب المعايير المطلوبة عليها

```bash
python -m testing.app
```

### 10. يمكنك عرض نتائج عملية الاختبار في الواجهة باستخدام الزر المخصص
