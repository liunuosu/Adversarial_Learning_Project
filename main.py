from model.ReplicationAdversary import ReplicationAdversary
from model.ReplicationAdversary_validation import ReplicationAdversary_validation
from model.GeneralAdversary import GeneralAdversary
from model.GeneralAdversary_validation import GeneralAdversary_validation
from model.MultipleAdversary import MultipleAdversary
from model.MultipleAdversary_validation import MultipleAdversary_validation
from model.MultipleFeaturesAdversary import MultipleFeaturesAdversary
from utils.metrics import DifferenceAverageOdds
from utils.SensitiveFeature0.dataloader0 import dataloader0
from utils.SensitiveFeature1.dataloader1 import dataloader1
from utils.SensitiveFeature2.dataloader2 import dataloader2
from sklearn.metrics import accuracy_score


def train_pipeline_biased():
    n_epoch = 50
    sens_feature = 'gender'
    data = dataloader0()
    x_train, x_test, y_train, y_test, numvars, categorical = data

    classification = y_train.columns.to_list()
    classification.remove(sens_feature)
    classification = classification[0]
    clf = ReplicationAdversary([sens_feature], num_epochs=n_epoch, batch_size=256,
                                   random_state=279, debias=False)
    clf.fit(x_train, y_train)
    print("\nTest Results\n")
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')


def train_pipeline_replication():
    print("Replication runs")

    n_epoch = 200
    sens_feature = 'gender'
    data = dataloader1()
    x_train, x_test, y_train, y_test, numvars, categorical = data
    classification = y_train.columns.to_list()
    classification.remove(sens_feature)
    classification = classification[0]

    n_epoch = 39
    adv = ReplicationAdversary([sens_feature], num_epochs=n_epoch, batch_size=256,
                               adversary_loss_weight=0.5, random_state=279, debias=True)
    n_epoch = 121
    gen = GeneralAdversary([sens_feature], num_epochs=n_epoch, batch_size=256,
                           adversary_loss_weight=30, random_state=29)

    adv.fit(x_train, y_train)
    print("\nTest Results replication adversary\n")
    y_pred = adv.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')

    gen.fit(x_train, y_train)
    print("\nTest Results general adversary\n")
    y_pred = gen.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')


def train_pipeline_validation():
    print("Validation set runs")

    n_epoch = 50
    sens_feature = 'gender'
    data = dataloader1(validation=True)
    x_train, x_validation, x_test, y_train, y_validation, y_test, numvars, categorical = data

    classification = y_train.columns.to_list()
    classification.remove(sens_feature)
    classification = classification[0]

    #clf = ReplicationAdversary_validation([sens_feature], num_epochs=n_epoch, batch_size=256,
     #                          random_state=279, debias=False)
    adv = ReplicationAdversary_validation([sens_feature], adversary_loss_weight=1.5, num_epochs=n_epoch, batch_size=256,
                               random_state=29, debias=True)
    gen = GeneralAdversary_validation([sens_feature], adversary_loss_weight=2, num_epochs=n_epoch, batch_size=256,
                                        random_state=25)
    mlt = MultipleAdversary_validation([sens_feature], adversary_loss_weight=0.01, num_epochs=n_epoch, batch_size=256,
                                        random_state=297)

    #clf.fit(x_train, y_train, x_validation, y_validation)
    #print("\nTest Results biased classifier\n")
    #y_pred = clf.predict(x_test)
    #acc = accuracy_score(y_test[classification], y_pred)
    #dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    #print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')

    adv.fit(x_train, y_train, x_validation, y_validation)
    print("\nTest Results replication adversary\n")
    y_pred = adv.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')

    gen.fit(x_train, y_train, x_validation, y_validation)
    print("\nTest Results general adversary\n")
    y_pred = gen.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')

    mlt.fit(x_train, y_train, x_validation, y_validation)
    print("\nTest Results multiple adversaries\n")
    y_pred = mlt.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')


def train_pipeline_multiple():
    print("Multiple sensitive feature runs")
    data = dataloader2()
    x_train, x_test, y_train, y_test, numvars, categorical = data
    n_epoch = 50
    sens_feature = 'gender'
    sens_feature2 = 'race'
    classification = y_train.columns.to_list()
    classification.remove(sens_feature)
    classification.remove(sens_feature2)
    classification = classification[0]

    clf_2feature = MultipleFeaturesAdversary([sens_feature], [sens_feature2], adversary_loss_weight=0.5,
                                             num_epochs=n_epoch, batch_size=256,
                                             random_state=279, debias=False)
    adv_2feature = MultipleFeaturesAdversary([sens_feature], [sens_feature2], adversary_loss_weight=0.5,
                                             num_epochs=n_epoch, batch_size=256,
                                             random_state=279, debias=True)
    mlt_2feature = MultipleFeaturesAdversary([sens_feature], [sens_feature2], adversary_loss_weight=0.1,
                                             num_epochs=n_epoch, batch_size=256,
                                             random_state=297, multiple=True)

    clf_2feature.fit(x_train, y_train)
    print("\nTest Results biased classifier\n")
    y_pred = clf_2feature.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')

    adv_2feature.fit(x_train, y_train)
    print("\nTest Results one replication adversary\n")
    y_pred = adv_2feature.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')

    mlt_2feature.fit(x_train, y_train)
    print("\nTest Results multiple adversaries\n")
    y_pred = mlt_2feature.predict(x_test)
    acc = accuracy_score(y_test[classification], y_pred)
    dao = DifferenceAverageOdds(y_pred, y_test, sens_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {acc}, \nDiff. in Average Odds: {dao}')


#train_pipeline_biased()
train_pipeline_replication()
#train_pipeline_validation()
#train_pipeline_multiple()