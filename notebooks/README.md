## 运行顺序：
1. `data_preprocessing.ipynb`
2. `rec_training.ipynb`
3. `benchmark.ipynb` 

`**_functions.ipynb`不是可运行的notebook，主要存放的是函数体
`**benchmark.ipynb`：实现了一个统一的、严格的Benchmark评测体系，用于比较不同样本归因/影响力方法在推荐模型上的表现。核心思想是：通过“反事实轨迹”指标（删除/插入训练样本后的模型效用变化）来评估哪种方法能更好地识别重要样本。