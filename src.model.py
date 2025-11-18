# src/model.py

import torch
import torch.nn as nn

# دالة مساعدة لتحميل الإعدادات (نفس الدالة في data_processor.py)
import yaml
def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

class KashfClassifier(nn.Module):
    """
    نموذج تصنيف بسيط يعتمد على طبقة تضمين وطبقة LSTM/GRU.
    """
    def __init__(self, config):
        super(KashfClassifier, self).__init__()
        
        # قراءة المعاملات من ملف الإعدادات
        vocab_size = config['PREPROCESSING']['VOCAB_SIZE']
        embedding_dim = config['PREPROCESSING']['EMBEDDING_DIM']
        max_len = config['PREPROCESSING']['MAX_SEQUENCE_LENGTH']
        
        # عدد الفئات المستهدفة (نفترض أنها ثنائية: 0 أو 1)
        num_classes = 1 
        
        # 1. طبقة التضمين (Embedding Layer)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=0 # تجاهل بادينغ
        )
        
        # 2. طبقة LSTM (أو GRU)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=64, # حجم الخلية المخفية
            num_layers=2,   # عدد الطبقات
            batch_first=True,
            bidirectional=True # معالجة النص من الاتجاهين
        )
        
        # 3. طبقة كثيفة (Dense Layer) وطبقة إسقاط (Dropout)
        self.dropout = nn.Dropout(0.5)
        # 64 * 2 (لأنها ثنائية الاتجاه)
        self.fc = nn.Linear(64 * 2, num_classes) 

    def forward(self, x):
        # x: (حجم الدفعة, أقصى طول)
        
        # 1. تمرير التضمين
        embedded = self.embedding(x)
        # embedded: (حجم الدفعة, أقصى طول, بُعد التضمين)
        
        # 2. تمرير الـ LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 3. استخدام ناتج آخر خطوة زمنية (يتم دمج الاتجاهين)
        # نأخذ الحالة المخفية للطبقة الأخيرة (hidden[-2:] يعني الاتجاهين)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # 4. تمرير الإسقاط والطبقة الكثيفة
        out = self.dropout(hidden)
        out = self.fc(out)
        
        # ناتج النموذج (يحتاج إلى دالة Sigmoid في دالة الخسارة للتصنيف الثنائي)
        return out

if __name__ == '__main__':
    # اختبار سريع لهيكل النموذج
    config = load_config()
    
    # يجب أن تكون مفاتيح الإعدادات مطابقة لما قدمناه في الخطوة 4
    config_adjusted = {
        'PREPROCESSING': {
            'VOCAB_SIZE': 10000,
            'EMBEDDING_DIM': 100,
            'MAX_SEQUENCE_LENGTH': 256
        }
    }
    
    model = KashfClassifier(config_adjusted)
    print("تم إنشاء النموذج بنجاح:")
    print(model)
