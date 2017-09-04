# crawled_data (Sources from  http://www.thesarcasmdetector.com/ )
  Twitter Message crawled by twitter_crawler.py
  Already Crawled Data:
  1. twitDB_regular.csv : non-sarcastic sentences
  2. twitDB_sarcasm.csv : sarcastic sentences

# preproc.py
  After crawling from Twitter,do
  ```
  python2 preproc.py
  ```
  then, we got
  ```
  sarcasm_data_proc.npy and nonsarc_data_proc.npy
  ```
  as the final dataset to run the program

# select_test.py
  split dataset into training  set and testing set
  ```
  python2 select_test.py
  ```

# label_sarcasm : twitter txt splitted into 3-folds for mannual labeling  

# GoogleNews-vectors-negative300.bin.gz: pre-trained word2vector download from https://code.google.com/archive/p/word2vec/
