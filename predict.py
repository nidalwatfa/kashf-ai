# src/predict.py

import torch
import torch.nn.functional as F
import argparse

# استيراد الوحدات الخاصة بنا
from model import KashfClassifier, load_config 

# (يجب إضافة وحدة الترميز (Tokenization) والـ Vectorization هنا)
# ...

def preprocess_input(text_input, config):
    """
    تطبيق نفس خطوات المعالجة المسبقة المستخدمة في التدريب.
    (مثال: التنظيف، الترميز، والبادينغ إلى أقصى طول)
    """
    max_len = config['PREPROCESSING']['MAX_SEQUENCE_LENGTH']
    
    # يجب تطبيق منطق الترميز (Tokenizer) الذي تم تدريبه على بيانات التدريب
    # **لغرض العرض، سنقوم بإنشاء مدخلات عشوائية مؤقتة**
    
    # يجب أن يتم تمرير قائمة النصوص إلى Tokenizer للحصول على مُتجهات (Vectors)
    # مثال: vectors = tokenizer.texts_to_sequences([text_input])
    
    # ناتج هذه الدالة يجب أن يكون مُتجه PyTorch جاهز للدخول إلى النموذج
    # مثال: return torch.tensor(padded_vectors).long()
    
    # بيانات عشوائية مؤقتة:
    dummy_input = torch.randint(1, config['PREPROCESSING']['VOCAB_SIZE'], (1, max_len)).long()
    return dummy_input


def predict_text(text_input):
    """
    تحميل النموذج المدرب وإجراء عملية التنبؤ.
    """
    # 1. تحميل الإعدادات
    config = load_config()
    global_config = config['GLOBAL_SETTINGS']
    
    # 2. إعداد الجهاز ومسار النموذج
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = global_config['OUTPUT_MODEL_DIR'] + global_config['MODEL_FILENAME']
    
    # 3. بناء النموذج وتحميل الأوزان المدربة
    model = KashfClassifier(config).to(device)
    
    try:
        # تحميل أوزان النموذج
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # وضع التقييم (Inference Mode)
    except FileNotFoundError:
        return f"خطأ: لم يتم العثور على النموذج المدرب في المسار: {model_path}"
    
    # 4. معالجة الإدخال
    input_tensor = preprocess_input(text_input, config).to(device)
    
    # 5. التنبؤ
    with torch.no_grad(): # إيقاف حساب التدرجات لتسريع التنبؤ
        output = model(input_tensor)
        
    # 6. تحويل الناتج إلى قيمة تصنيف
    # (Sigmoid لتحويل الناتج إلى احتمالية بين 0 و 1، ثم تقريبها)
    probability = torch.sigmoid(output).item()
    prediction = 1 if probability >= 0.5 else 0
    
    return {
        "text": text_input,
        "prediction": "كشف/إيجابي" if prediction == 1 else "غير كشف/سلبي",
        "probability": f"{probability:.4f}"
    }

if __name__ == '__main__':
    # إضافة وسيط لتمرير النص المراد فحصه من سطر الأوامر
    parser = argparse.ArgumentParser(description="Kashf-AI Prediction Script")
    parser.add_argument('text', type=str, help='النص المراد تقييمه.')
    args = parser.parse_args()
    
    result = predict_text(args.text)
    
    print("\n--- نتيجة نظام كشف (Kashf-AI) ---")
    print(f"النص المدخل: {result['text']}")
    print(f"التصنيف المحتمل: {result['prediction']}")
    print(f"درجة الاحتمال: {result['probability']}")
    print("----------------------------------\n")

# طريقة التشغيل من سطر الأوامر:
# python src/predict.py "هذا نص تجريبي لتقييم النموذج"
