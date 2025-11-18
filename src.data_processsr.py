
# src/data_processor.py

# ... (بقية الاستيرادات: torch, pandas, sklearn, pyarabic, nltk, yaml, pickle)
# يتم إزالة استيراد: from tensorflow.keras.preprocessing.text import Tokenizer
# يتم إزالة استيراد: from tensorflow.keras.preprocessing.sequence import pad_sequences

# الاستيراد الجديد من torchtext
from torchtext.legacy import data
from torchtext.vocab import Vocab

# (يجب أن تبقى دوال clean_text_pipeline، normalize_arabic، remove_stopwords، save_tokenizer كما هي)

# -------------------------------------------------------------

def load_and_preprocess_data(config):
    """
    تحميل البيانات، تطبيق المعالجة المسبقة، وبناء الـ Field (المرممز) باستخدام torchtext.
    """
    
    # ... (الجزء الأول: تحميل البيانات وتطبيق التنظيف)
    # df = pd.read_csv(...)
    # df['text_content'] = df['text_content'].apply(clean_text_pipeline)
    
    # 1. تعريف وظيفة الـ Tokenizer (سنستخدم دالة split() البسيطة)
    def tokenizer_func(text):
        # هنا يمكنك استخدام nltk.word_tokenize أو split() بعد التنظيف
        return text.split()

    # 2. تعريف الـ Field (يحتوي على تعليمات المعالجة)
    MAX_LEN = config['PREPROCESSING']['MAX_SEQUENCE_LENGTH']
    
    TEXT = data.Field(
        tokenize=tokenizer_func,
        lower=True, # تحويل إلى أحرف صغيرة (لغير العربية)
        include_lengths=False, # لا نحتاج لأطوال النصوص في هذا النموذج
        batch_first=True,
        fix_length=MAX_LEN
    )
    
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    
    # 3. تقسيم البيانات إلى مجموعات (نستخدم pandas/sklearn أولاً لضمان التحكم)
    # X_train_text, X_test_text, y_train_series, y_test_series = train_test_split(...)
    
    # 4. تحويل البيانات إلى تنسيق (Dataset) من torchtext
    # (هنا يتم تطبيق الـ Tokenization والـ Padding تلقائياً عبر Field)
    
    # يجب أن تكون البيانات في شكل قائمة من القوائم (List of Lists) ليتم التعامل معها من قبل torchtext
    fields = [('text', TEXT), ('label', LABEL)]
    
    # إنشاء قائمة الأمثلة (Examples)
    train_examples = []
    for text, label in zip(X_train_text, y_train_series):
        train_examples.append(data.Example.fromlist([text, label], fields))

    test_examples = []
    for text, label in zip(X_test_text, y_test_series):
        test_examples.append(data.Example.fromlist([text, label], fields))
        
    train_data = data.Dataset(train_examples, fields)
    test_data = data.Dataset(test_examples, fields)

    # 5. بناء قاموس المفردات (Vocabulary)
    # يستخدم بيانات التدريب فقط لحساب تكرارات الكلمات
    TEXT.build_vocab(train_data, max_size=config['PREPROCESSING']['VOCAB_SIZE'], min_freq=2)
    
    # 6. حفظ قاموس المفردات للوصول إلى المفردات في نموذج التنبؤ
    vocab_path = config['GLOBAL_SETTINGS']['OUTPUT_MODEL_DIR'] + 'vocab.pkl'
    save_tokenizer(TEXT.vocab, vocab_path)
    
    # 7. إنشاء أدوات تحميل البيانات (Data Loaders) - تستخدم في train.py
    BATCH_SIZE = config['TRAINING_HYPERPARAMETERS']['BATCH_SIZE']
    
    train_iterator = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=torch.device('cpu'))
    test_iterator = data.BucketIterator(test_data, batch_size=BATCH_SIZE, device=torch.device('cpu'))
    
    # في هذه الحالة، يتم تمرير الـ Iterators ومعلومات المفردات إلى train.py
    
    return train_iterator, test_iterator, len(TEXT.vocab)
