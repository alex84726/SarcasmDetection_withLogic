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
  then, we got sarcasm_data_proc.npy and nonsarc_data_proc.npy as the final
  dataset to run the program

# label_sarcasm : twitter txt splict into 3-folds for mannual labeling  
