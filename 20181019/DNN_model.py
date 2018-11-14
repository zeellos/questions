import tensorflow as tf
import numpy as np

# setting initial parameters
FEATURE_NUMBER = 800
LABEL_COLUMN = '801'
FEATURE_NAMES = [str(i) for i in range(0,FEATURE_NUMBER)]
CSV_COLUMN = [str(i) for i in range(0,FEATURE_NUMBER)]
CSV_COLUMN.append(LABEL_COLUMN)
CSV_COLUMN_DEFAULTS = [[0.0] for i in range(0,FEATURE_NUMBER)]
CSV_COLUMN_DEFAULTS.append([0])
UNUSED_COLUMNS = set()
train_path = 'training.csv'
test_path = 'testing.csv'
test_path_labels = 'testing_label_only.csv'
model_dir = 'output'

def parse_csv(rows_string_tensor):
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMN, columns))
    for col in UNUSED_COLUMNS:
        features.pop(col)
    return features 

def input_fn_train(filenames,
                        num_epochs=None,
                        shuffle=True,
                        skip_header_lines=0,
                        batch_size=200,
                        modeTrainEval=True):
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        filename_dataset = filename_dataset.shuffle(len(filenames))
    dataset = filename_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(skip_header_lines))
    dataset = dataset.map(parse_csv)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    features, labels = features, features.pop(LABEL_COLUMN)
    if not modeTrainEval:
        return features, None
    return features, labels


def main(argv):
    batch_size = 100
    train_steps = 10000
    eval_steps = 10
    num_epochs = 1

    my_feature_columns = []
    for key in FEATURE_NAMES:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    hidden_units=[1024, 512, 256] 

    estimator = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=hidden_units,
        model_dir=model_dir,
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01,l1_regularization_strength=0.001))

    train_input = lambda: input_fn_train([train_path],batch_size=batch_size,skip_header_lines=0)
    train_spec = tf.estimator.TrainSpec(train_input,max_steps=train_steps)

    eval_input = lambda: input_fn_train([test_path],num_epochs=num_epochs,shuffle=False,skip_header_lines=0)
    eval_spec = tf.estimator.EvalSpec(eval_input,steps=eval_steps,throttle_secs=100)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    '''
    #estimator.train(train_input, steps=train_steps)
    #metrics = estimator.evaluate(input_fn=lambda:input_fn_predict(test_x, test_y, batch_size), steps=10)
    #predictions = estimator.predict(input_fn=lambda:input_fn_train(test_x,num_epochs = 1))
    
    predict_input = lambda: input_fn_train([test_path],num_epochs=num_epochs,shuffle=False,skip_header_lines=0,modeTrainEval = False)
    predictions = estimator.predict(predict_input)
    
    test_y = genfromtxt(test_path_labels, delimiter=',')

    pred_list_np = np.array(list(predictions))
    pred_np = np.zeros(pred_list_np.shape[0])
    i = 0
    for pred in pred_list_np:
        pred_np[i] = int(pred['class_ids'])
        i = i + 1
    TP, FP, TN, FN = perf_measure(test_y, pred_np)
    #np.savetxt("predictions.csv", pred_np, delimiter=",")
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    sensitivity =  TP/(FN + TP)
    specificity  =  TN/(FP + TN)
    print (TP, FP, TN, FN)
    print ("sensitivy: ", sensitivity)
    print ("specificity: ", specificity)
    print ("accuracy: ", accuracy)
    '''
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
