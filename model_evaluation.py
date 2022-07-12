from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from building_and_training_NN import model
from data_preprocesing import videos_train, labels_train

prediction = model.predict(videos_train)
prediction_true = np.argmax(labels_train, axis=1).tolist()
prediction = np.argmax(prediction, axis=1).tolist()
print(multilabel_confusion_matrix(prediction_true, prediction))
print(accuracy_score(prediction_true, prediction))
print(model.summary())
