import math
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
class adversary_model(tf.Module):
    def __init__(self, seed3):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        self.c = tf.Variable(tf.constant(1.0), name='c')

        # uniform initializer
        self.W1 = tf.Variable(tf.random.normal(shape=(3, 1), seed=self.seed3), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

    # function to use adversary
    def forward(self, classifier_logits, true_income):
        s = tf.sigmoid((1 + tf.abs(self.c)) * classifier_logits)
        adversary_logits = tf.matmul(tf.concat([s, s * true_income, s * (1.0 - true_income)], axis=1), self.W1) + self.b1
        return adversary_logits


# Class for second adversary
class multiple_model(tf.Module):
    # first testing without dropout
    def __init__(self, n_features, seed2, seed4):
        super(multiple_model, self).__init__()
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


# class for running classifer or classifier + adversary
class MultipleAdversary_validation(BaseEstimator, ClassifierMixin):
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
        self.adv_model = None
        self.mlt_model = None

        # setting our seeds for weight initialization
        rng = check_random_state(self.random_state)

        ii32 = np.iinfo(np.int32)
        # we accidentally did all our tuning with seed 3 instead of seed 2 (:
        self.s1, self.s2, self.s3, self.s4 = rng.randint(ii32.min, ii32.max, size=4)

        # I don't know what the line below does, but when I remove it, our results get worse
        tf.random.set_seed(self.random_state)

    def fit(self, x_train, y_train, x_validation, y_validation):
        """Train the classifier (and adversary if ``debias == True``) with the
        given training data.
        Args:
            x_train (pandas.DataFrame): Training samples.
            y_train (array-like): Training labels.
        Returns:
            self
            :param y_validation: Validation labels
            :param x_validation: Validation samples
        """

        # changing X to the right datatype
        if scipy.sparse.issparse(x_train):
            x_train = x_train.todense()
        if scipy.sparse.issparse(x_validation):
            x_validation = x_validation.todense()

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
        starter_learning_rate_classifier = 0.0005
        # 0.1 worked well for constant learning_rate_adversary with InversetimeDecay for classifier
        starter_learning_rate_adversary = 0.0005

        starter_learning_rate_multiple = 0.0005

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

        self.adv_model = adversary_model(seed3=self.s3)
        adversary_opt = tf.optimizers.Adam(starter_learning_rate_adversary)

        self.mlt_model = multiple_model(n_features=n_features, seed2=self.s2, seed4= self.s4)
        multiple_opt = tf.optimizers.Adam(starter_learning_rate_multiple)

        normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

        # Modify Patience here
        patience = 10

        early_stopper = EarlyStopper(patience=patience, min_delta=0.001)
        checkpoint_adv = tf.train.Checkpoint(model=self.clf_model)
        manager_adv = tf.train.CheckpointManager(checkpoint_adv, directory='./checkpoints_adv',
                                                 max_to_keep=patience)
        earlystop_clf, save_epoch = 100, 0

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
                with tf.GradientTape() as tape_adv:
                    classifier_pred, classifier_logits = self.clf_model.forward(
                        batch_features)  # variables of CLF_model need to be watched from tape1 also
                    # runs adversary model with classifier_logit as input
                    pred_protected_attributes_logits = self.adv_model.forward(
                        classifier_logits, batch_labels)
                    loss_adv = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits))
                adv_grad = tape_adv.gradient(loss_adv, classifier_vars)

                # Gradient WlA2
                with tf.GradientTape() as tape_mlt:
                    classifier_pred, classifier_logits = self.clf_model.forward(
                        batch_features)
                    pred_protected_attributes_logits_mlt = self.mlt_model.forward(
                        classifier_logits, batch_labels)
                    loss_mlt = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits_mlt))
                mlt_grad = tape_mlt.gradient(loss_mlt, classifier_vars)

                for _, (grad, var) in enumerate(zip(classifier_grad, self.clf_model.trainable_variables)):

                    # all lines are just calculating the descent direction (fancy formula paper) - DOES NOT WORK
                    # normalizing gradients adversaries w.r.t W
                    adv1_grad = normalize(adv_grad[_])
                    adv2_grad = normalize(mlt_grad[_])

                    # Compute perpendicular gradient a2 - proj_a2(a1). (a2 = adv2, a1 = a1 model)
                    perp_adv2_grad = adv2_grad - (tf.reduce_sum(adv2_grad * adv1_grad) * adv1_grad)
                    perp_adv2_grad = normalize(perp_adv2_grad)

                    # Correct projection, first term
                    grad -= tf.reduce_sum(grad * adv1_grad) * adv1_grad
                    # finalize projection
                    grad -= tf.reduce_sum(grad * perp_adv2_grad) * perp_adv2_grad
                    grad -= self.adversary_loss_weight * (adv1_grad + adv2_grad)
                    descent_direction.append((grad, var))

                classifier_opt.apply_gradients(descent_direction)

                # Update adversary model
                with tf.GradientTape() as tape_adv:
                    pred_protected_attributes_logits = self.adv_model.forward(
                       classifier_logits, batch_labels)
                    loss_adv = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits))
                gradients_adv = tape_adv.gradient(loss_adv, self.adv_model.trainable_variables)
                adversary_opt.apply_gradients(zip(gradients_adv, self.adv_model.trainable_variables))

                # Update multiple model
                with tf.GradientTape() as tape_mlt:
                    pred_protected_attributes_logits_mlt = self.mlt_model.forward(
                       classifier_logits, batch_labels)
                    loss_mlt = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                logits=pred_protected_attributes_logits_mlt))
                gradients_mlt = tape_mlt.gradient(loss_mlt, self.mlt_model.trainable_variables)
                multiple_opt.apply_gradients(zip(gradients_mlt, self.mlt_model.trainable_variables))

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

                    batch_dao, p1, p2, p1_n, p2_n = compute_DAO(y_pred_all, y_train, feature, predict, 1, 0, [0, 1])
                    clf_accuracy = accuracy(predict_class(pred_labels_all),
                                            y_train.drop(columns=self.prot_attr))  # tf.reduce_mean(clf_acc)

                    pred_protected_attributes_logits_all = self.adv_model.forward(pred_logits_all, y_train.drop(
                        columns=self.prot_attr))

                    loss_adv = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[self.prot_attr].astype('float32'),
                                                                logits=pred_protected_attributes_logits_all))
                    adv_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all)),
                                            y_train[self.prot_attr])

                    pred_protected_attributes_logits_all_mlt = self.mlt_model.forward(pred_logits_all, y_train.drop(
                        columns=self.prot_attr))
                    loss_mlt = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[self.prot_attr].astype('float32'),
                                                                logits=pred_protected_attributes_logits_all_mlt))
                    mlt_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all_mlt)),
                                            y_train[self.prot_attr])

                    # Computation of validation set results

                    pred_labels_all_validation, pred_logits_all_validation = self.clf_model.forward(
                        x_validation.astype('float32'))
                    y_pred_all_validation = np.array(predict_class(pred_labels_all_validation))
                    batch_dao_validation, p1_val, p2_val, p1_n_val, p2_n_val = compute_DAO(y_pred_all_validation,
                                                                                           y_validation, feature,
                                                                                           predict, 1, 0,
                                                                                           [0, 1])
                    clf_accuracy_validation = accuracy(predict_class(pred_labels_all_validation),
                                                       y_validation.drop(columns=self.prot_attr))

                    pred_protected_attributes_logits_all_validation = self.adv_model.forward(
                        pred_logits_all_validation,
                        y_validation.drop(columns=self.prot_attr))
                    adv_accuracy_validation = accuracy(
                        predict_class(tf.sigmoid(pred_protected_attributes_logits_all_validation)),
                        y_validation[self.prot_attr])


                    print(
                        "(TRAINING  ) epoch %d; classifier accuracy: %f; "
                        "DAO: %f; FPRf: %f; FPRm: %f; FNRf: %f ; FNRm: %f adversary accuracy: %f;" % (
                            epoch + 1, clf_accuracy, batch_dao, p1, p2, p1_n, p2_n, adv_accuracy))

                    # Print validation results
                    print(
                        "(VALIDATION) epoch %d; classifier accuracy: %f; DAO %f; FPRf: %f; FPRm: %f; FNRf: "
                        "%f " "; FNRm: %f; adversary accuracy: %f"
                        % (epoch + 1, clf_accuracy_validation, batch_dao_validation, p1_val, p2_val, p1_n_val,
                           p2_n_val, adv_accuracy_validation))


                    earlystop_clf = loss_classifier
                    save_epoch = epoch

                    if early_stopper.new_minimum_dao(batch_dao_validation):
                        # METRICS FROM TRAINING DATA ARE TAKEN
                        manager_adv.save()
                        min_loss_adv = loss_adv
                        min_adv_acc = adv_accuracy
                        min_loss_clf = loss_classifier
                        min_clf_acc = clf_accuracy
                        min_batch_dao = batch_dao
                        min_FPRf = p1
                        min_FPRm = p2
                        min_FNRf = p1_n
                        min_FNRm = p2_n

                    # Early stop based on minimum batch_dao of validation set
            if early_stopper.early_stop_adv(earlystop_clf, save_epoch, clf_accuracy_validation,
                                            batch_dao_validation) or (epoch + 1) == self.num_epochs:
                print(
                    "(Adversarial Debiasing) epoch %d; classifier accuracy: %f; "
                    "DAO: %f; FPRf: %f; FPRm: %f; FNRf: %f ; FNRm: %f; adversary accuracy: %f" % (
                         early_stopper.get_epoch() + 1, min_clf_acc, min_batch_dao, min_FPRf, min_FPRm,
                         min_FNRf, min_FNRm, min_adv_acc))
                checkpoint_adv.restore(manager_adv.latest_checkpoint)
                break
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

    # Can ask Liu if you have any questions, but def fit is more important than these functions
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


def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    check_equal = tf.cast(y_pred == y, tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val


def predict_class(y_pred, thresh=0.5):
    # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
    return tf.cast(y_pred > thresh, tf.float32)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.epoch = 0
        self.clf_accuracy = 0
        self.DAO = 100
        self.accuracy_threshold = 0.8
        self.epoch_threshold = 10

    def early_stop(self, validation_loss, epoch, clf_accuracy, DAO):
        # if validation_loss < self.min_validation_loss:
        if clf_accuracy > self.clf_accuracy:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.epoch = epoch
            self.clf_accuracy = clf_accuracy
            self.DAO = DAO

        elif clf_accuracy < (self.clf_accuracy + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_adv(self, validation_loss, epoch, clf_accuracy, DAO):
        # Skip the first few epochs since they have low dao
        if DAO < self.DAO and clf_accuracy > 0.80:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.epoch = epoch
            self.clf_accuracy = clf_accuracy
            self.DAO = DAO

        # Delta is added to add minimum difference to count as a 'strike'
        elif DAO > (self.DAO + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def new_minimum(self, clf_accuracy):
        if clf_accuracy > self.clf_accuracy:
            return True
        return False

    def new_minimum_dao(self, dao):
        if dao < self.DAO:
            return True
        return False

    def get_epoch(self):
        return self.epoch

    def get_clf(self):
        return self.min_validation_loss

    def get_clf_accuracy(self):
        return self.clf_accuracy

    def get_DAO(self):
        return self.DAO


# computing DAO
def compute_DAO(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,
                                                                  labels=labels).ravel()

    n1 = FP_unpriv + TN_unpriv
    n2 = FP_priv + TN_priv
    p1 = FP_unpriv / n1
    p2 = FP_priv / n2

    pooled_p = (n1 * p1 + n2 * p2) / (n1 + n2)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    z_value = (p1 - p2) / (se + np.finfo(np.float32).tiny)

    p_value = scipy.stats.norm.sf(abs(z_value)) * 2

    n1 = FN_unpriv + TP_unpriv
    n2 = FN_priv + TP_priv
    p1_n = FN_unpriv / n1
    p2_n = FN_priv / n2

    pooled_p = (n1 * p1_n + n2 * p2_n) / (n1 + n2)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    z_value = (p1_n - p2_n) / (se + np.finfo(np.float32).tiny)

    p_value1 = scipy.stats.norm.sf(abs(z_value)) * 2

    return 0.5 * (abs(FP_unpriv / (FP_unpriv + TN_unpriv) - FP_priv / (FP_priv + TN_priv)) + abs(
        TP_unpriv / (TP_unpriv + FN_unpriv) - TP_priv / (TP_priv + FN_priv))), p1, p2, p1_n, p2_n