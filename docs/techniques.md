# 技术

### TF-IDF向量化器（TfidfVectorizer）

scikit-learn提供的TF-IDF向量化器是一种将文本数据转换为数值向量表示的常用技术，在自然语言处理中有着广泛的应用。它通过计算词语在文档中的词频（Term Frequency）和逆文档频率（Inverse Document Frequency）来衡量词语的重要性，并将其转化为数值特征，从而使计算机能够更好地理解和处理文本数据。

<aside>
💡

- 词频（TF）： 一个词语在一个文档中出现的次数。词频越高，通常认为该词语在该文档中越重要。
- 逆文档频率（IDF）： 一个词语在整个语料库中出现的文档数的倒数的对数。IDF值越高，说明该词语在整个语料库中越少见，越具有区分度。
- TF-IDF： TF和IDF的乘积，综合考虑了词语在文档中的局部重要性和在整个语料库中的全局重要性。
</aside>

- 工作原理：
    - 分词： 将文本分割成单个词语。
    - 构建词典： 将所有文档中的词语构建成一个词典。
    - 计算TF-IDF： 对于每个文档中的每个词语，计算其TF-IDF值。
    - 构建向量： 将每个文档的TF-IDF值组成一个向量，即该文档的向量表示。
- 优点：
    - 简单易懂： 概念清晰，计算简单。
    - 效果较好： 在很多文本分类、聚类等任务中表现良好。
    - 可解释性强： TF-IDF值可以直观地反映词语的重要性。
- 缺点：
    - 无法捕捉语义： 只考虑词频和文档频率，无法理解词语之间的语义关系。
    - 对停用词敏感： 停用词（如“的”、“是”等）的TF-IDF值可能很高，需要进行预处理。
- 应用场景：
    - 文本分类： 将文本分为不同的类别，如情感分类、新闻分类等。
    - 文本聚类： 将相似的文本文档聚在一起。
    - 信息检索： 在搜索引擎中对搜索结果进行排序。
    - 关键词提取： 提取文本中的关键词。
- 输入参数：
    - **`ngram_range`:** 指定要考虑的 n-gram 范围。例如，`(1, 1)` 表示只考虑 unigrams (单个词)，`(1, 2)` 表示考虑 unigrams 和 bigrams (两个词的组合)，`(1, 3)` 表示考虑 unigrams、bigrams 和 trigrams (三个词的组合)。
    - **`max_df`:** 忽略文档频率高于给定阈值的词。这可以帮助去除一些常见的、不具有区分性的词。
    - **`min_df`:** 忽略文档频率低于给定阈值的词。这可以帮助去除一些非常罕见的、可能不重要的词。
    - **`max_features`:** 指定要保留的最大特征数量。这可以帮助减少特征空间的维度。

### 多项式朴素贝叶斯分类器（MultinomialNB）

参数调优：多项式朴素贝叶斯（MultinomialNB）主要有一个参数可以调整：**`alpha` (平滑参数):**  `alpha` 是一个平滑参数，用于处理零概率问题。当 `alpha` 为 0 时，如果一个特征在训练集中没有出现过，那么在测试集中遇到该特征时，模型会将其概率计算为 0，导致预测错误。设置一个非零的 `alpha` 值可以避免这种情况。通常，`alpha` 的取值范围在 0 到 1 之间，默认值为 1 (拉普拉斯平滑)。较小的 `alpha` 值表示更少的平滑，而较大的 `alpha` 值表示更多的平滑。

### 网格搜索交叉验证模型调参器（GridSearchCV）

GridSearchCV 是 scikit-learn 中一个强大的工具，用于在机器学习模型中自动寻找最佳参数组合。它使用的方法是**网格搜索 + 交叉验证**： 它通过遍历所有可能的参数组合（形成一个网格），并使用交叉验证来评估每个组合的性能，从而找到最优参数。通过找到最适合数据的参数，可以显著提升模型的准确性、泛化能力等。

```python
class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
```

- 主要输入参数：
    - estimator：使用的模型(估计器)
    - param_grid：一个用于指定每个参数的取值范围的字典，以分类器的参数名称作为键（Key），要尝试的参数为值（value）。也可是此类字典组成的列表
    - scoring（=str, callable, list, tuple or dict, default=None）：用于评估模型性能的指标，比如accuracy、f1-score等，具体使用方法见：
        
        https://scikit-learn.org/dev/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
        
    - cv: 交叉验证的折数，默认为 None，表示使用默认的 3 折交叉验证。
    - n_jobs: 并行任务数，-1 表示使用所有 CPU 内核。
    - verbose: 控制输出信息的详细程度。
    - refit: 是否使用最佳参数重新训练模型。
- 其他输入参数：
    - iid: 是否假设所有样本的概率分布一致，默认为 True。
    - return_train_score: 是否返回训练集上的得分。
    - pre_dispatch: 并行任务的调度策略。
- 输出：GridSearchCV 在完成参数搜索和交叉验证后，会返回一个包含大量信息的 GridSearchCV 对象。这个对象可以通过其属性来访问各种结果，其中包含：
    - best_params_： 字典形式，包含了表现最佳的参数组合。
    - best_score_： float类型，表示最佳参数组合对应的交叉验证平均得分。
    - best_estimator_： estimator类型，使用最佳参数训练得到的模型。
    - cv_results_： 字典形式，包含了所有参数组合的详细结果，包括训练得分、测试得分、参数设置等。
    
    访问GridSearchCV 对象实例：
    
    ```python
    from sklearn.model_selection import GridSearchCV
    
    # ... (假设你已经创建了一个 GridSearchCV 对象 grid)
    grid = GridSearchCV(estimator, parameters, n_jobs=-1, verbose=1)
    
    # 打印最佳参数
    print(grid.best_params_)
    
    # 打印最佳得分
    print(grid.best_score_)
    
    # 使用最佳模型进行预测
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 访问所有结果
    cv_results = grid.cv_results_
    print(cv_results.keys())  # 查看所有的键
    print(cv_results['mean_test_score'])  # 打印平均测试得分
    
    ```
    

### 管道（Pipeline）

Pipeline（管道） 在机器学习中是一个非常有用的概念，由 scikit-learn提供。它可以将数据预处理、特征工程和模型训练等多个步骤串联封装起来，形成一个完整的机器学习流程。

> 使用Pipeline封装流程有以下优点：
> 
> - 简化代码：数据预处理过程是很繁杂凌乱的，pipeline 可以直接将步骤封装成完整的工作流，避免了代码重复。
> - 减少Bug：避免在建模过程中出现步骤遗漏。
> 更易于生产/复制：将建好的模型应用到实际数据，或者大规模部署到不同的数据集时，过程会非常繁琐，我们不需要在处理很多的问题，但Pipline可以帮助我们省略这些重复的过程，直接调用fit和predict来对所有算法模型进行训练和预测。
> - 简化模型验证过程：比如将pipeline 和交叉验证结合有助于防止来自测试数据的统计数据泄露到交叉验证的训练模型中。或者与网格搜索（Grid Search）结合，快速遍历所有参数的结果。
> * [原文链接](https://blog.csdn.net/WHYbeHERE/article/details/125074001)
> 

```python
class sklearn.pipeline.Pipeline(steps, *, memory=None, verbose=False)
```

- 输入参数：Pipeline 由多个步骤（step）组成，每个步骤是一个元组，包含一个包含名称的键（key）和一个包含模型（estimator）的值（value）。估计器可以是转换器（transformer）或模型（estimator）。
    - 转换器： 用于对数据进行转换，比如标准化、归一化、特征选择等。
    - 模型： 用于进行预测或分类。
- 输出：使用pipeline()函数就可以得到管道输出，输出的管道可以作为模型（estimator）使用

### 其他函数

apply()：在Pandas中，apply(）常 用于对 DataFrame 的每一行或每一列进行自定义 操作。

split()：Python 字符串处理中非常常用的一 个方法，可以方便地将字符串分割成列表，从而方便 后续的处理。 

```python
string.split(separator=None, maxsplit=-1)
```

- string：要分割的字符串。
- separator：分隔符，默认为所有的空字符（包 括空格、换行符等）。
- maxsplit：最多分割的次数，默认为-1，表示不 限制分割次数。

join()：将列表中的元素用指定的分隔符连接成 一个字符串。

lower()： 将字符串中的所有大写字母转换为小写字母，保存到一个新字符串并返回，原始字符串保持不变。
fit() ： 在 Scikit-learn 中，几乎所有的机器学习模型都有 fit() 方法。fit() 函数就是用来将训练数据输入到模型中，让模型学习数据中的模式和规律。 在训练过程中，模型会自动调整内部参数，以最小化损失函数。fit() 函数就是负责这个参数调整的过程。

- fit() 函数的输入：
    - 训练数据： 通常是一个包含特征（特征矩阵）和标签（目标变量）的数据集。
    - 其他参数： 不同模型的 fit() 函数可能还有一些其他的参数，比如学习率、迭代次数等。
- fit() 函数的输出： fit() 函数返回一个训练好的模型对象，这个对象可以用来进行预测。

## 保存和加载模型

**1. 使用 `pickle`:**

`pickle` 是 Python 的标准库，可以用来序列化和反序列化 Python 对象，包括 scikit-learn 模型。

- **保存模型:**

```python
import pickle

# ... (训练模型代码) ...

# 保存模型到文件
filename = 'model.pkl'
pickle.dump(best_model, open(filename, 'wb'))

```

- **加载模型:**

```python
import pickle

# 加载模型
loaded_model = pickle.load(open('model.pkl', 'rb'))

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)

```

**2. 使用 `joblib` (推荐):**

`joblib` 是专门为序列化 NumPy 数组而设计的，因此在保存包含大型 NumPy 数组的 scikit-learn 模型时，`joblib` 通常比 `pickle` 更高效。

- **保存模型:**

```python
from joblib import dump, load

# ... (训练模型代码) ...

# 保存模型到文件
filename = 'model.joblib'
dump(best_model, filename)

```

- **加载模型:**

```python
from joblib import dump, load

# 加载模型
loaded_model = load('model.joblib')

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)

```