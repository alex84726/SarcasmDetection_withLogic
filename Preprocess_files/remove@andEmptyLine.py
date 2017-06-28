if __name__ == '__main__':
    lines = open(
        '../../data/english/positive_without_hashtag.txt',
        'r',
        encoding='utf-8',
        errors='ignores').read().split('\n')
    corpus = []
    for line in lines:
        word = line.split(' ')
        words = [w for w in word if ('@' not in w) and ('http' not in w)]
        sentence = ' '.join(words)
        if sentence != '':
            corpus.append(sentence)
    outfile = open(
        '../../data/english/positive_without_hashtag_parse.txt',
        'w',
        encoding='utf-8')
    for line in corpus:
        outfile.write(line)
        outfile.write('\n')
    outfile.close()
    lines = open(
        '../../data/english/negative_without_hashtag.txt',
        'r',
        encoding='utf-8',
        errors='ignores').read().split('\n')
    corpus = []
    for line in lines:
        word = line.split(' ')
        words = [w for w in word if ('@' not in w) and ('http' not in w)]
        sentence = ' '.join(words)
        if sentence != '':
            corpus.append(sentence)
    outfile = open(
        '../../data/english/negative_without_hashtag_parse.txt',
        'w',
        encoding='utf-8')
    for line in corpus:
        outfile.write(line)
        outfile.write('\n')
    outfile.close()
