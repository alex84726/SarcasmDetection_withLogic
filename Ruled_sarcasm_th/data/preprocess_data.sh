# preprocess raw data
python2.7 preprocess_stsa.py ./raw ./w2v/GoogleNews-vectors-negative300.bin
# extract rule features
python2.7 logicnn_features.py ./stsa.sarcasm.p 
