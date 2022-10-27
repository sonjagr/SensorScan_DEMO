# Created by Sonja Grönroos in August 2022
#
# Copyright (c) 2021 CERN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from config import *
from classes.ModelHandler import *
from utility.dataset_helpers import (box_to_labels, open_db, combine_datasets,
                                     create_cnn_dataset, format_new_anomalous_dataset, format_normal_dataset, combine_normal_datasets,
                                     process_anomalous_df_to_numpy)
from utility.fs import (check_new_test_data, create_dir,
                        filter_db, open_filter_db, str_to_dt)
random.seed(42)

# parser = argparse.ArgumentParser()
# parser.add_argument("--Mode", type=str, help="(input) 'train', 'continue' or 'test'.",
#                    required=True)
# parser.add_argument("--modelName", type=str, help="(input) name of the model to process.",
#                    required=True)
# args = parser.parse_args()

# mode = args.Mode
# model_name = args.modelName

model_handler = Modelhandler()

validation_db = pd.read_pickle(VAL_DB_LOC)
anomalous_validation_db = validation_db[validation_db.x != "nan"]
normal_validation_db = validation_db[validation_db.x == "nan"]

newest_val_date = anomalous_validation_db.ValDate.max()

mode = "retrain"
model_name = "class_cnn_clean_fl_epoch_20_2022-07-26"

model_date = str_to_dt(model_name.split("_")[-1])

filepath = os.path.join(MODELS_DIR_LOC, "class_cnn_clean_fl_epoch_{epoch:02d}_loss_{val_loss:.2f}_%s" % newest_val_date)

def update_datasets():
    print("\n--- Checking for new validated data ---")
    old_anom_train_db = open_db("TRAIN_DATABASE")
    newest_train_ds_date = old_anom_train_db.Date.unique().max()
    newest_train_ds_date_list = old_anom_train_db.Date.unique().tolist()
    if newest_val_date <= newest_train_ds_date:
        newest_train_ds_date_list = []
        print("The data sets are up-to-date \n")
    else:
        print(
            "New data has been added to the validation database that is not on newest training data created on %s. Newest data was gathered on %s." % (
                newest_train_ds_date, newest_val_date))
        while True:
            add = input("Will you add the new data to old datasets and create new dataset files (yes/no)? \n")
            if add == "yes":
                old_anom_test_db = open_db("TEST_DATABASE")
                old_anom_val_db = open_db("VAL_DATABASE")

                (new_anom_train_db, new_anom_val_db, new_anom_test_db, new_date, new_data_files) = format_new_anomalous_dataset(
                    anomalous_validation_db, newest_train_ds_date)
                combine_datasets(old_anom_train_db, new_anom_train_db, "TRAIN")
                combine_datasets(old_anom_val_db, new_anom_val_db, "VAL")
                combine_datasets(old_anom_test_db, new_anom_test_db, "TEST")

                (new_norm_train_db, new_norm_val_db, new_norm_test_db) = format_normal_dataset(normal_validation_db, newest_train_ds_date)
                combine_normal_datasets(new_norm_train_db, "TRAIN")
                combine_normal_datasets(new_norm_val_db, "VAL")
                combine_normal_datasets(new_norm_test_db, "TEST")

                newest_train_ds_date = new_anom_train_db.Date.unique().max()
                newest_train_ds_date_list = new_anom_train_db.Date.unique().tolist()
                break
            elif add == "no":
                print("\nData sets are not updated.")
                break
            print("Try again")
    return newest_train_ds_date, newest_train_ds_date_list

newest_train_ds_date, newest_train_ds_date_list = update_datasets()

while True:
    if not os.path.isdir(os.path.join(MODELS_DIR_LOC, model_name)):
        print("Model does not exist. Try again.")
        break

    if mode.casefold() != "test".casefold():
        print()
        print("--- Initiating training program ---")
        print("You have chosen model in ", model_name, "\n")

        train_db = open_db("TRAIN_DATABASE")
        val_db = open_db("VAL_DATABASE")

        if newest_train_ds_date == model_date and mode.casefold() == 'retrain'.casefold():
            print("The model has already been trained with the newest training data. Consider running again with 'mode = continue' or with an older model.")
            print('Exiting.')
            exit()

        elif newest_train_ds_date == model_date and mode.casefold() == 'continue'.casefold():
            print( "The model has been trained with the newest training data. You will continue to train with the same data.")

        elif model_date < newest_train_ds_date:
            print("--- Select training data ---")
            print("The model was previously trained on data gathered until %s. New data has been added on %s. \n"% (model_date, newest_train_ds_date_list))
            while True:
                data_date = input( "Press enter to use all newest data, input 'model' to use same data as before, or input date of data until which you want to include data [YYYY-mm-dd]. \n")
                if data_date == "":
                    print("Using all data.")
                    break
                elif data_date == 'model':
                    print("Using the same data than previously to train the model")
                    train_db = filter_db(train_db, model_date)
                    val_db = filter_db(val_db, model_date)
                    break
                elif str_to_dt(data_date) == model_date and mode.casefold() == 'retrain'.casefold():
                    print("You chose the same training set that the model has already been trained on for retraining.")
                    print("Consider running again with 'mode = continue' or with an older model.")
                    print("Exiting.")
                    exit()
                else:
                    try:
                        data_date = str_to_dt(data_date)
                        train_db = filter_db(train_db, data_date)
                        val_db = filter_db(val_db, data_date)
                        break
                    except:
                        print("Error in the date you inputted. Try again.")
                        exit()

        print()
        print("--- Loading model ---")
        model = model_handler.loadClassifierModel(model_name=model_name, filepath=filepath)
        if mode.casefold() == 'retrain'.casefold():
            print("As you have chosen to retrain model, weights will be re-initialized. \n")
            model_handler.reinit_model()

        date_from_train_ds = train_db.Date.unique().max()
        filepath = os.path.join(MODELS_DIR_LOC,"class_cnn_clean_fl_epoch_{epoch:02d}_loss_{val_loss:.2f}_%s" % date_from_train_ds)

        print("New model will be saved as %s." % filepath)
        print()
        model_handler.set_mode(mode)

        while True:
            filename = input("Give name of train parameter file. To use same as before, type in 'original'. \n")
            if filename == 'original':
                model_handler.loadTrainParameters_json(filename='trainParams_original.json')
                break
            else:
                try:
                    model_handler.loadTrainParameters_json(filename=filename)
                    break
                except:
                    print('Try again.')
        print()
        while True:
            start = input("Press enter to start training. Type in x to exit. \n")
            if start == "":
                print("--- Training started ---")
                model_handler.init_training(filepath=filepath, train_db=train_db, val_db=val_db)
            if start == "x":
                exit()
            else:
                print("Try again")
        break


    elif mode.casefold() == "test".casefold():
        print()
        print("--- Initiating testing program ---")
        print("You have chosen to test model in ", model_name, "\n")
        model_handler.set_classification_th(0.1)
        model = model_handler.loadClassifierModel(model_name=model_name, filepath=filepath)
        print("Default test data for the model was collected until %s." % model_date)
        test_ds = open_db("TEST_DATABASE")  # model_date
        val_ds = open_db("VAL_DATABASE")
        if model_date < newest_train_ds_date:
            print("Newer test data exists. \n")
            while True:
                print("Press enter if you will use default test data or input date of newer test data you want to use [YYYY-mm-dd].")
                newers = test_ds[test_ds.Date > model_date].Date.unique().tolist()
                newer_date = input("Newer dates are: %s \n" % [str(i) for i in newers])
                if newer_date.strip():
                    try:
                        newer_date = str_to_dt(newer_date)
                        test_ds = filter_db(test_ds, newer_date)  # newer date
                        val_ds = filter_db(val_ds, newer_date)
                        print("Using testing data collected until %s. \n" % newer_date)
                        break
                    except:
                        print("Data with that date does not exist. Try again. \n")
                else:
                    print("Using default database. \n")
                    test_ds = filter_db(test_ds, model_date)
                    val_ds = filter_db(val_ds, model_date)
                    break
        else:
            print("Using default database. \n")
        model_handler.create_testing_save_loc(os.path.join(MODELS_DIR_LOC, model_name))
        model_handler.loadTrainParameters_json(filename='trainParams_original.json')
        model_handler.init_testing(test_ds, val_ds)
        break
    else:
        print("Incorrect mode. Try again.")
        break
