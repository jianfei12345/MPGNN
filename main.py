import GraphClassification_src
import numpy as np

#performance
dataset = [('lipo', 'regression', 'stratified'), ('freesolv', 'regression', 'stratified'), ('hiv', 'classification', 'stratified'), ('bace_classification', 'classification', 'stratified'), ('bbbp', 'classification', 'stratified'), ('delaney', 'regression', 'stratified')]
result = GraphClassification_src.MP_run(dataset)

#multi-task performance
result_multitask = GraphClassification_src.multi_task_n_times(5, Dataset='tox21', model_name=GraphClassification_src.Graph_Class, mode='classification')

np.save('MP_result.npy', result)
np.save('MP_result_multitask.npy', result_multitask)


#parameter analysis
batch_size = [8, 16, 32, 64, 128]
learning_rate = [0.0001, 0.001, 0.005, 0.01, 0.05]
result_batch_size_classification = []
result_batch_size_regression = []
result_learning_rate_classification = []
result_learning_rate_regression = []
for i in batch_size:
    temp = GraphClassification_src.run_n_times(Dataset='bbbp', model=GraphClassification_src.Graph_Class, mode='classification', n=10, batch_size=i)
    result_batch_size_classification.append(temp)
    temp1 = GraphClassification_src.run_n_times(Dataset='delaney', model=GraphClassification_src.Graph_Class, mode='regression', n=10, batch_size=i)
    result_batch_size_regression.append(temp1)

for i in learning_rate:
    temp = GraphClassification_src.run_n_times(Dataset='bbbp', model=GraphClassification_src.Graph_Class, mode='classification', learning_rate=i, n=10)
    result_learning_rate_classification.append(temp)
    temp1 = GraphClassification_src.run_n_times(Dataset='delaney', model=GraphClassification_src.Graph_Class, mode='regression', learning_rate=i, n=10)
    result_learning_rate_regression.append(temp1)

np.save('MP_result_batch_size_classification.npy', result_batch_size_classification)
np.save('MP_result_batch_size_regression.npy', result_batch_size_regression)
np.save('MP_result_learning_rate_classification.npy', result_learning_rate_classification)
np.save('MP_result_learning_rate_regression.npy', result_learning_rate_regression)



#ablation analysis
model_variants = [GraphClassification_src.Graph_Class, GraphClassification_src.Graph_Class_ablation1, GraphClassification_src.Graph_Class_ablation2, GraphClassification_src.Graph_Class_ablation3, GraphClassification_src.Graph_Class_ablation4, GraphClassification_src.Graph_Class_ablation5]
result_ablation_classification = {}
result_ablation_regression = {}
for i in model_variants:
    result_ablation_classification[str(i)] = GraphClassification_src.run_n_times(Dataset='bbbp', model=i, mode='classification', n=10)
    result_ablation_regression[str(i)] = GraphClassification_src.run_n_times(Dataset='delaney', model=i, mode='regression', n=10)

np.save('MP_result_ablation_classification.npy', result_ablation_classification)
np.save('MP_result_ablation_regression.npy', result_ablation_regression)








