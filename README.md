
SemEval-2017 Task 5
Fine-Grained Sentiment Analysis on Financial Microblogs and News

http://alt.qcri.org/semeval2017/task5/

Usage example:

-- Cross validation
python main.py --train_file Microblog_Trainingdata.json --mode cv --subtask 1
python main.py --train_file Headline_Trainingdata.json --mode cv --subtask 2

-- Prediction
python main.py --train_file Microblog_Trainingdata.json --test_file Microblogs_Testdata.json --mode predict --subtask 1 --regressor svm
python main.py --train_file Headline_Trainingdata.json --test_file Headlines_Testdata.json --mode predict --subtask 2 --regressor rnn

Train and test data files can be downloaded from SemEval-2017 site
