# src/data_processor.py

# إضافة الاستيراد اللازم
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle # لحفظ وتحميل الـ Tokenizer
# ... (بقية الاستيرادات والدوال السابقة)

# ... (دوال load_config و clean_text_pipeline)

def save_tokenizer(tokenizer, path='models/tokenizer.pkl'):
    """ حفظ كائن Tokenizer باستخدام pickle """
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"تم حفظ الـ Tokenizer في: {path}")

def load_tokenizer(path='models/tokenizer.pkl'):
    """ تحميل كائن Tokenizer من المسار المحدد """
    try:
        with open(path, 'rb') as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        print(f"خطأ: لم يتم العثور على ملف الـ Tokenizer في المسار: {path}")
        return None

def load_and_preprocess_data(config):
    """
    تحميل البيانات، التنظيف، إنشاء الـ Tokenizer، والتقسيم.
    """
    # ... (الجزء الأول: تحميل البيانات وتطبيق التنظيف)
    # df = pd.read_csv(...)
    # df['text'] = df['text'].apply(clean_text_pipeline)
    
    # 1. تقسيم البيانات الأولية (قبل الترميز)
    X = df['text']
    y = df[config['DATA_CONFIGURATION']['TARGET_COLUMN']]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['TRAINING_HYPERPARAMETERS']['TEST_SIZE'], 
        random_state=config['GLOBAL_SETTINGS']['RANDOM_SEED'],
        stratify=y 
    )
    
    # 2. إنشاء وتدريب الـ Tokenizer على بيانات التدريب فقط!
    # يجب أن يتم حفظ الـ Tokenizer في مجلد models/ لكي يتم تجاهله بـ .gitignore
    vocab_size = config['PREPROCESSING']['VOCAB_SIZE']
    
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<unk>" # رمز للكلمات غير الموجودة في المفردات
    )
    # تدريب الـ Tokenizer على نصوص التدريب
    tokenizer.fit_on_texts(X_train_text)
    
    # 3. حفظ الـ Tokenizer
    tokenizer_path = config['GLOBAL_SETTINGS']['OUTPUT_MODEL_DIR'] + 'tokenizer.pkl'
    save_tokenizer(tokenizer, tokenizer_path)
    
    # 4. تحويل النصوص إلى متسلسلات رقمية (Sequences)
    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_text)
    
    # 5. البادينغ (Padding)
    max_len = config['PREPROCESSING']['MAX_SEQUENCE_LENGTH']
    
    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=max_len,
        padding='post', # إضافة الأصفار بعد النص
        truncating='post' # قطع النص من النهاية إذا كان أطول من max_len
    )
    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )
    
    # 6. التحويل إلى مُتجهات PyTorch
    X_train_tensor = torch.tensor(X_train_padded).long()
    X_test_tensor = torch.tensor(X_test_padded).long()
    
    # تحويل الفئة المستهدفة إلى متجهات PyTorch
    y_train_tensor = torch.tensor(y_train.values).float().unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values).float().unsqueeze(1)

    print(f"تم ترميز وتجهيز البيانات بنجاح.")
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
