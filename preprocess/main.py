import sys, pickle, string, os
import numpy as np
DATA_PATH  = '../../data/'
ENG_DATA_PATH = '../../data/english/'
CHI_DATA_PATH = '../../data/chinese/'

# only for python3
translator = str.maketrans({key: None for key in string.punctuation})
def main():


    """
    dict_w, MAXLEN= make_dictionary(
        ENG_DATA_PATH+'posproc.npy',
        ENG_DATA_PATH+'negproc.npy')

    dummy_id = len(dict_w)

    dict_path = DATA_PATH+'dictionary_english_v{}.p'.format(len(dict_w))
    pickle.dump((dict_w), open(dict_path, 'wb'), protocol=2)

    write_onehot_pickle(dict_w, dummy_id, MAXLEN, \
                        ENG_DATA_PATH+'posproc.npy', \
                        DATA_PATH+'tweet_positive_dummy{}.npy'.format(dummy_id))
    write_onehot_pickle(dict_w, dummy_id, MAXLEN, \
                        ENG_DATA_PATH+'negproc.npy', \
                        DATA_PATH+'tweet_negative_dummy{}.npy'.format(dummy_id))

    """
    dict_w, MAXLEN = make_dictionary(
        CHI_DATA_PATH+'irony_ch_output_new.txt',
        CHI_DATA_PATH+'NTU.seq.space.20000.txt') 
    dummy_id = len(dict_w)

    dict_path = DATA_PATH+'dictionary_chinese_v{}.p'.format(len(dict_w))
    pickle.dump((dict_w), open(dict_path, 'wb'), protocol=2)

    write_onehot_pickle(dict_w, dummy_id, MAXLEN, \
                        CHI_DATA_PATH+'irony_ch_output_new.txt', \
                        DATA_PATH+'NTUIrony.p')
    write_onehot_pickle(dict_w, dummy_id, MAXLEN, \
                        CHI_DATA_PATH+'NTU.seq.space.20000.txt', \
                        DATA_PATH+'PTT_NTU_20000.p')


def make_dictionary(positive_path, negative_path):
    """
    positive_path: path for sarcastic file
    negative_path: path for non-sarcastic file

    return
        dictionary
        dummy_id: integer, (represent padded symbol)
        max_length: max_length for both positve and negative files
    """
    dictionary = {}
    idex = 0; line_num = 0; max_len = 0

    for path in [positive_path, negative_path]:
        _, file_extension = os.path.splitext(path)
        if file_extension == '.txt':
            fi = open(path, 'r')
        elif file_extension == '.npy':
            fi = np.load(open(path, 'rb'))
            fi = [x.decode('UTF-8') for x in fi]
        for line in fi:
            line = line.translate(translator).lower().split()
            if len(line) > max_len: max_len = len(line)
            for w in line:
                if w not in dictionary:
                    dictionary[w] = idex
                    idex += 1
            line_num += 1
        try:
            fi.close()
        except: pass # fi is not file I/O
        print('Number of sentence in {}:  {}'.format(path, line_num))
        line_num = 0

    print('Number of unique word: {}'.format(len(dictionary)))
    print('Number of max length: {}'.format(max_len))
    return dictionary, max_len

def extend_dictionary(dict, path, max_len):
    """
    dict: original dictionary
    path: new file need to be extract words
    max_len: lastest max_len

    return
        dictionary
        dummy_id: integer, (represent padded symbol)
        max_length: integer, new max_length
    """
    idex = start_index = len(dict_w) - 1
    line_num = 0
    with open(path, 'r') as fi:
        for line in fi:
            line = line.translate(translator).lower().split()
            if len(line) > max_len: max_len = len(line)
            for w in line:
                if w not in dictionary:
                    dictionary[w] = idex
                    idex += 1
            line_num += 1
    print('Number of sentence in {}:  {}'.format(path, line_num))
    print('Add {} new word '.format(index - start_index))
    print('New max_len: {}'.format(max_len))
    return dictionary, max_len


def write_onehot_pickle(dictionary, dummy_id, max_len, input_path, output_path):
    book = []
    _, file_extension = os.path.splitext(input_path)
    if file_extension == '.txt':
        fi = open(input_path, 'r')
    elif file_extension == '.npy':
        fi = np.load(open(input_path, 'rb'))
        fi = [x.decode('UTF-8') for x in fi]
    for line in fi:
        #line = line.strip('?/,.!@&*()\"\':/<>==+-[]').lower().split()
        line = line.translate(translator).lower().split()
        line = list(map(lambda x : dictionary[x], line))
        padding = [dummy_id for _ in range(max_len - len(line))]

        line = line + padding
        book.append(line)

    try:
        fi.close()
    except: pass # fi is not file I/O
    book = np.asarray(book, dtype=np.int32)
    with open(output_path, 'wb') as fo:
        np.save(fo, book, fix_imports=True)


if __name__ == '__main__':
    main()
