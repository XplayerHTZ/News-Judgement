# 加载并使用模型
from joblib import load

loaded_best_model = load('best_model.joblib')
new_text = ["This is a sample news article to test the model."] #  示例新文本
new_text_transformed = loaded_best_model.named_steps['tfidf'].transform(new_text) # 使用训练时的tfidf转换器
prediction = loaded_best_model.predict(new_text_transformed)
print(prediction) # 输出预测结果