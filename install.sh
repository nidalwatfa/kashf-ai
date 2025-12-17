#!/data/data/com.termux/files/usr/bin/bash

# تحديث الحزم وتثبيت الأدوات الأساسية
pkg update -y
pkg upgrade -y
pkg install python git ffmpeg -y

# استنساخ المشروع من GitHub
git clone https://github.com/nidalwatfa/kashf-ai
cd kashf-ai

# تعديل requirements.txt لتجاوز مكتبات غير مدعومة
sed -i '/torch/d' requirements.txt
sed -i '/torchtext/d' requirements.txt

# تثبيت المكتبات المتاحة
pip install -r requirements.txt

echo "✅ تم التثبيت بنجاح! يمكنك الآن تشغيل التطبيق باستخدام:"
echo "python app.py --input test.mp4"

