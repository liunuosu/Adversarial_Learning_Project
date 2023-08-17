import numpy as np
import scipy.special
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state
import tensorflow as tf


# class for classifier
class classifier_model(tf.Module):
    def __init__(self, features, seed1):
        super(classifier_model, self).__init__()
        self.features = features
        self.seed1 = seed1

        # no hidden layer
        # uniform initializer
        self.W1 = tf.Variable(tf.random.uniform(shape=(self.features, 1), seed=self.seed1), name='W1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

    # function to use classifier
    def forward(self, x):
        # no hidden layer
        classifier_logits = tf.matmul(x, self.W1) + self.b1
        classifier_pred = tf.sigmoid(classifier_logits)
        return classifier_pred, classifier_logits


# class for adversary
class adversary1_model(tf.Module):
    # first testing without dropout
    def __init__(self, n_features, seed3, seed5):
        super(adversary1_model, self).__init__()
        self.n_features = n_features

        self.seed3 = seed3
        self.seed5 = seed5

        # uniform initializer
        self.W1 = tf.Variable(tf.random.normal(shape=(2, self.n_features), seed=self.seed3), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

        self.W2 = tf.Variable(tf.random.normal(shape=(self.n_features, 1), seed=self.seed5), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=1), name='b2')

    # function to use adversary
    def forward(self, classifier_logits, true_income):
        input_hidden_layer = tf.concat([classifier_logits, true_income], axis=1)
        hidden_layer = tf.nn.relu(tf.matmul(input_hidden_layer, self.W1)) + self.b1
        # hidden_layer = tf.nn.dropout(hidden_layer, rate=0, seed=self.seed4)
        adversary_logits = tf.matmul(hidden_layer, self.W2) + self.b2
        adversary_pred = tf.sigmoid(adversary_logits)

        return adversary_logits


# Class for second adversary
class adversary2_model(tf.Module):
    def __init__(self, n_features, seed2, seed4):
        super(adversary2_model, self).__init__()
        self.n_features = n_features

        self.seed2 = seed2
        self.seed4 = seed4

        # uniform initializer
        self.W1 = tf.Variable(tf.random.normal(shape=(2, self.n_features), seed=self.seed2), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

        self.W2 = tf.Variable(tf.random.normal(shape=(self.n_features, 1), seed=self.seed4), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=1), name='b2')

    # function to use adversary
    def forward(self, classifier_logits, true_income):
        input_hidden_layer = tf.concat([classifier_logits, true_income], axis=1)
        hidden_layer = tf.nn.relu(tf.matmul(input_hidden_layer, self.W1)) + self.b1
        # hidden_layer = tf.nn.dropout(hidden_layer, rate=0, seed=self.seed4)
        adversary_logits = tf.matmul(hidden_layer, self.W2) + self.b2
        # print(tf.shape(adversary_logits))
        adversary_pred = tf.sigmoid(adversary_logits)

        return adversary_logits


# class for running classifier or classifier + adversary
class MultipleAdversary(BaseEstimator, ClassifierMixin):
    def __init__(self, prot_attr=None, scope_name='classifier',
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=256, random_state=None):

        r"""
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the debiasing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
            scope_name (str, optional): TensorFlow "variable_scope" name for the
                entire model (classifier and adversary).
            adversary_loss_weight (float or ``None``, optional): If ``None``,
                this will use the suggestion from the paper:
                :math:`\alpha = \sqrt(global_step)` with inverse time decay on
                the learning rate. Otherwise, it uses the provided coefficient
                with exponential learning rate decay.
            num_epochs (int, optional): Number of epochs for which to train.
            batch_size (int, optional): Size of mini-batch for training.
            debias (bool, optional): If ``False``, learn a classifier without an
                adversary.
            random_state (int or numpy.RandomState): Seed of pseudo-random number generator
            for shuffling data and seeding weights.
        """

        self.prot_attr = prot_attr
        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # here so we don't get orange lines
        self.classes_ = None
        self.clf_model = None
        self.adv1_model = None
        self.adv2_model = None

        # setting our seeds for weight initialization
        rng = check_random_state(self.random_state)

        ii32 = np.iinfo(np.int32)
        # we accidentally did all our tuning with seed 3 instead of seed 2 (:
        self.s1, self.s2, self.s3, self.s4, self.s5 = rng.randint(ii32.min, ii32.max, size=5)

        # I don't know what the line below does, but when I remove it, our results get worse
        tf.random.set_seed(self.random_state)

    def fit(self, x_train, y_train):
        """Train the classifier (and adversary if ``debias == True``) with the
        given training data.
        Args:
            x_train (pandas.DataFrame): Training samples.
            y_train (array-like): Training labels.
        Returns:
            self
        """

        # changing X to the right datatype
        if scipy.sparse.issparse(x_train):
            x_train = x_train.todense()

        # protected_attribute = gender, datatype = pandas.core.frame.DataFrame
        protected_attribute = y_train[self.prot_attr]
        # dependent_variable_classifier = income, datatype = pandas.core.frame.DataFrame
        dependent_variable_classifier = y_train.drop(columns=self.prot_attr)

        # self.classes_ is used in def predict, we believe it is just the value 2
        # perhaps change his def predict with the one Liu wrote after the draft deadline
        self.classes_ = np.unique(dependent_variable_classifier)

        # dimensions data
        num_train_samples, n_features = x_train.shape

        self.clf_model = classifier_model(features=n_features, seed1=self.s1)

        # learning rate = η in paper. The code originally used exponential decay, decay rate 0.96 and decay steps 1000.
        # The paper suggest that eta should be proportional to 1/t (InverseTimeDecay).
        # Paper suggested using InverseTimeDecay in combination with adversarial_loss_weight (alpha) = sqrt(t).
        # We get good results when using InverseTimeDecay with alpha = 0.16, but not with alpha = sqrt(t).
        # I also tested different learning rate for classifier and adversary with inverseTimeDecay
        # I also tested constant learning rates for both classifier and adversary and got good results
        starter_learning_rate_classifier = 0.002
        # 0.1 worked well for constant learning_rate_adversary with InversetimeDecay for classifier
        starter_learning_rate_adversary = 0.001

        starter_learning_rate_multiple = 0.001

        # used for decayed learning rate
        learning_rate_classifier = tf.keras.optimizers.schedules.InverseTimeDecay(starter_learning_rate_classifier,
                                                                       decay_steps=1025, decay_rate=0.66,
                                                                       staircase=True)
        # defining optimizer with decayed learning rate
        # classifier_opt = tf.optimizers.Adam(learning_rate_classifier)
        # defining optimizer with constant learning rate
        classifier_opt = tf.optimizers.Adam(starter_learning_rate_classifier)

        # makes a tensor with trainable variables classifier
        classifier_vars = [var for var in self.clf_model.trainable_variables]

        self.adv1_model = adversary1_model(n_features=n_features, seed3=self.s3, seed5=self.s5)
        adversary_opt = tf.optimizers.Adam(starter_learning_rate_adversary)

        self.adv2_model = adversary2_model(n_features=n_features, seed2=self.s2, seed4=self.s4)
        multiple_opt = tf.optimizers.Adam(starter_learning_rate_multiple)

        normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

        for epoch in range(self.num_epochs):
            shuffled_ids = [i for i in range(num_train_samples)]
            for i in range(num_train_samples // self.batch_size):
                batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                batch_features = x_train[batch_ids].astype('float32')
                batch_labels = np.reshape(dependent_variable_classifier.iloc[batch_ids], [-1, 1]).astype('float32')
                batch_protected_attributes = np.reshape(protected_attribute.iloc[batch_ids], [-1, 1]).astype(
                    'float32')

                descent_direction = []

                # Compute the perpendicular of gradient of mlt model on gradient of adv model

                # Implement gradient formula, get classifier gradient Wlp
                with tf.GradientTape() as tape:
                    classifier_pred, classifier_logits = self.clf_model.forward(batch_features)
                    loss_classifier = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=classifier_logits))
                classifier_grad = tape.gradient(loss_classifier, classifier_vars)

                # Gradient WLA1
                with tf.GradientTape() as tape_adv1:
                    classifier_pred, classifier_logits = self.clf_model.forward(
                        batch_features)  # variables of CLF_model need to be watched from tape1 also
                    # runs adversary model with classifier_logit as input
                    pred_protected_attributes_logits = self.adv1_model.forward(
                        classifier_logits, batch_labels)
                    loss_adv1 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits))
                adv1_grad = tape_adv1.gradient(loss_adv1, classifier_vars)

                # Gradient WlA2
                with tf.GradientTape() as tape_adv2:
                    classifier_pred, classifier_logits = self.clf_model.forward(
                        batch_features)
                    pred_protected_attributes_logits_mlt = self.adv2_model.forward(
                        classifier_logits, batch_labels)
                    loss_adv2 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits_mlt))
                adv2_grad = tape_adv2.gradient(loss_adv2, classifier_vars)

                for _, (grad, var) in enumerate(zip(classifier_grad, self.clf_model.trainable_variables)):
                    # all lines are just calculating the descent direction (fancy formula paper)
                    # normalizing gradients adversaries w.r.t W
                    # these might be completely wrong
                    adv1_grad = normalize(adv1_grad)
                    adv2_grad = normalize(adv2_grad)

                    # Compute perpendicular gradient a2 - proj_a2(a1). (a2 = adv2, a1 = a1 model)
                    perp_adv2_grad = adv2_grad - tf.reduce_sum(adv2_grad * adv1_grad) * adv1_grad
                    perp_adv2_grad = normalize(perp_adv2_grad)

                    # Correct projection, first term
                    grad -= tf.reduce_sum(grad * adv1_grad) * adv1_grad
                    # finalize projection
                    grad -= tf.reduce_sum(grad * perp_adv2_grad) * perp_adv2_grad
                    grad -= self.adversary_loss_weight * (adv1_grad + perp_adv2_grad)
                    descent_direction.append((grad, var))

                classifier_opt.apply_gradients(descent_direction)

                # Update adversary model
                with tf.GradientTape() as tape_adv1:
                    pred_protected_attributes_logits = self.adv1_model.forward(
                       classifier_logits, batch_labels)
                    loss_adv1 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits))
                gradients_adv1 = tape_adv1.gradient(loss_adv1, self.adv1_model.trainable_variables)
                adversary_opt.apply_gradients(zip(gradients_adv1, self.adv1_model.trainable_variables))

                # Update multiple model
                with tf.GradientTape() as tape_adv2:
                    pred_protected_attributes_logits_mlt = self.adv2_model.forward(
                       classifier_logits, batch_labels)
                    loss_adv2 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits_mlt))
                gradients_adv2 = tape_adv2.gradient(loss_adv2, self.adv2_model.trainable_variables)
                multiple_opt.apply_gradients(zip(gradients_adv2, self.adv2_model.trainable_variables))

                # Printing of the results
                if i % (num_train_samples // self.batch_size - 1) == 0 and i != 0:
                    feature = 'gender'
                    predict = 'income'

                    # training data metrics
                    pred_labels_all, pred_logits_all = self.clf_model.forward(x_train.astype('float32'))
                    y_pred_all = np.array(predict_class(pred_labels_all))

                    loss_classifier = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=y_train.drop(columns=self.prot_attr).astype('float32'),
                            logits=pred_logits_all))

                    batch_dao = compute_DAO(y_pred_all, y_train, feature, predict, 1, 0, [0, 1])
                    clf_accuracy = accuracy(predict_class(pred_labels_all),
                                            y_train.drop(columns=self.prot_attr))  # tf.reduce_mean(clf_acc)

                    pred_protected_attributes_logits_all = self.adv1_model.forward(pred_logits_all, y_train.drop(
                        columns=self.prot_attr))

                    loss_adv1 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[self.prot_attr].astype('float32'),
                                                                logits=pred_protected_attributes_logits_all))
                    adv_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all)),
                                            y_train[self.prot_attr])

                    pred_protected_attributes_logits_all_mlt = self.adv2_model.forward(pred_logits_all, y_train.drop(
                        columns=self.prot_attr))
                    loss_adv2 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[self.prot_attr].astype('float32'),
                                                                logits=pred_protected_attributes_logits_all_mlt))
                    adv2_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all_mlt)),
                                            y_train[self.prot_attr])

                    print(
                        "(Multiple debiasing) epoch %d; iter: %d; classifier loss: %f; "
                        "adversarial loss: %f; classifier accuracy: %f; adversary accuracy: %f; "
                        "multiple accuracy: %f; DAO: %f" % (
                            epoch, i, loss_classifier, loss_adv1, clf_accuracy, adv_accuracy, adv2_accuracy, batch_dao))
        return self


    def decision_function(self, X):
        """Soft prediction scores.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Confidence scores per (sample, class) combination. In
            the binary case, confidence score for ``self.classes_[1]`` where >0
            means this class would be predicted.
        """
        if scipy.sparse.issparse(X):
            X = X.todense()
        num_test_samples = X.shape[0]
        # n_classes = len(self.classes_)

        # if n_classes == 2:
        #    n_classes = 1 # lgtm [py/unused-local-variable]

        self.clf_model.dropout = 0
        samples_covered = 0
        pred_labels_list = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = X[batch_ids]
            pred_labels, pred_logits = self.clf_model.forward(batch_features.astype("float32"))

            pred_labels_list += pred_labels.numpy().tolist()
            samples_covered += len(batch_features)

        scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        return scores.ravel() if scores.shape[1] == 1 else scores


    def predict(self, X):
        """Predict class labels for the given samples.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            if X.shape[0] == 1:
                indices = (scores > 0.5).astype(int).reshape((-1,))
            else:
                indices = (scores > 0.5).astype(int).reshape((-1,))
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]


# needed for accuracy for printing metrics epochs
def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    check_equal = tf.cast(y_pred == y, tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val


# needed for DAO for printing metrics epochs
def predict_class(y_pred, thresh=0.5):
    # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
    return tf.cast(y_pred > thresh, tf.float32)


# computing DAO each epoch
def compute_DAO(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,
                                                                  labels=labels).ravel()

    return 0.5 * (abs(FP_unpriv / (FP_unpriv + TN_unpriv) - FP_priv / (FP_priv + TN_priv)) + abs(
        TP_unpriv / (TP_unpriv + FN_unpriv) - TP_priv / (TP_priv + FN_priv)))