#!/bin/bash

# جعل السكربت قابل للتنفيذ (اختياري إذا شغلته يدوي)
chmod +x install.sh

# إنشاء بيئة افتراضية
python -m venv venv

# تفعيل البيئة
source venv/bin/activate   # على لينكس/ماك
# .\venv\Scripts\activate   # على ويندوز (غيّر السطر حسب نظامك)

# ترقية pip
pip install --upgrade pip

# تثبيت كل المتطلبات
pip install -r requirements.txt

# تأكيد الانتهاء
echo "=================================="
echo "تم تثبيت كل الحزم بنجاح!"
echo "البيئة جاهزة - اكتب 'deactivate' للخروج لاحقًا"
echo "=================================="
