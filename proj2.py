if __name__ == '__main__':
    import warnings
    from sklearn.ensemble import RandomForestClassifier
    warnings.filterwarnings("ignore")
    import gensim
    from pprint import pprint
    import gensim.corpora as corpora
    from gensim.models import CoherenceModel

    # Plotting tools
    import pyLDAvis
    import pyLDAvis.gensim_models

    import matplotlib.pyplot as plt
    # % matplotlib inline
    import numpy as np
    import os
    import random

    filePath = r"jyxstxtqj_downcc.com/"  # 文件夹路径
    fileList = os.listdir(filePath)

    testFilePath = "test.txt"
    trainFilePath = "train.txt"
    allFilePath = "all.txt"
    open(testFilePath, 'w').close()
    open(trainFilePath, 'w').close()
    open(allFilePath, 'w').close()

    testFile = open(testFilePath, 'a', encoding='utf-8')
    trainFile = open(trainFilePath, 'a', encoding='utf-8')
    allFile = open(allFilePath, 'a', encoding='utf-8')


    def not_empty(s):
        return s and s.strip() and len(s) > 200


    trainlist = []
    testlist = []
    selected_all = []
    for file in fileList:
        f = open(os.path.join(filePath, file), encoding='gb18030')
        if file in ["倚天屠龙记.txt", "笑傲江湖.txt", "天龙八部.txt", "射雕英雄传.txt", "神雕侠侣.txt"]:
            print(file)
            paralist = []
            selected = []
            fullText = f.read()
            paralist = fullText.split("\n")
            # print(len(paralist))
            paralist = list(filter(not_empty, paralist))
            # print(len(paralist))
            selected = random.sample(paralist, 200)
            random.shuffle(selected)
            selected_all.extend(selected)
            # print(selected)
            trainlist.extend(selected[0:180])
            testlist.extend(selected[180:200])

    M = 1000
    # print(len(selectedparalist))
    # print(selectedparalist[0])


    for para in trainlist:
        trainFile.write(para + "\n\n\n\n")
    for para in testlist:
        testFile.write(para + "\n\n\n\n")
    for para in selected_all:
        allFile.write(para + "\n\n\n\n")

    trainFile.close()
    testFile.close()
    allFile.close()

    import jieba.posseg as psg
    import jieba


    def remove_blank_space(contents):
        contents_new = map(lambda s: s.replace(' ', ''), contents)
        return list(contents_new)


    def is_chinese_words(words):
        for word in words:
            if word >= u'\u4e00' and word <= u'\u9fa5':
                continue
            else:
                return False
        return True


    def word_filter(seg_list):
        # filter_list = []
        # #  进行词性过滤，选择名词
        # for seg in seg_list:
        #     word = seg.word
        #     flag = seg.flag
        #     if not flag.startswith('n'):
        #         continue
        #     filter_list.append(word)
        return seg_list


    def cut_words(contents):
        # cut_contents = map(lambda s: list(jieba.cut(s)), contents)
        # cut_contents = map(lambda s: list(psg.cut(s)), contents)
        cut_contents = [char for char in contents]
        # cut_contents = map(word_filter, list(cut_contents))
        # return list(cut_contents)
        return cut_contents


    def drop_stopwords(contents):
        # 初始化获取停用词表
        stop=open('cn_stopwords.txt', encoding='utf-8')
        # stop_me=open('/data/GAOzihan/nlpcourse/NLP_3/data/word_deal/stop_one_mx.txt', encoding='utf-8')
        # key_words = open('./data/word_deal/key_words.txt', encoding='utf-8')
        # 分割停用词/自定义停用词/关键词
        stop_words = stop.read().split("\n")
        # stop_me_words = stop_me.read().split("\n")
        stpwrdlst = []
        stpwrdlst.extend(list(stop_words))
        # stpwrdlst.extend(list(stop_me_words))
        # key_words = key_words.read().split("\n")
        # 定义返回后的结果
        contents_new = []
        # 遍历处理数据
        for line in contents:
            line_clean = []
            for word in line:
                if (word in stop_words):  # and word not inkey_words:      or word in stop_me_words
                    continue
                if is_chinese_words(word):
                    line_clean.append(word)
            contents_new.append(line_clean)
        return contents_new, stpwrdlst


    def read_text_file(url):
        dream_text = open(url, 'r+', encoding='utf-8')
        return list(filter(not_empty, dream_text.read().split("\n\n\n\n")))
        # return list(dream_text.read().split("\n\n"))


    train_text = read_text_file('train.txt')
    test_text = read_text_file('test.txt')
    all_text = read_text_file('all.txt')  # 去除空格
    train_data = remove_blank_space(train_text)
    test_data = remove_blank_space(test_text)
    all_data = remove_blank_space(all_text)
    #  获取分词结果，并且用过滤器只保留名词
    train_data = cut_words(train_data)
    test_data = cut_words(test_data)
    all_data = cut_words(all_data)
    #  去除停用词, 只保留中文字符
    train_data, stpwrdlst = drop_stopwords(train_data)
    test_data, stpwrdlst = drop_stopwords(test_data)
    all_data, stpwrdlst = drop_stopwords(all_data)
    print("len(train_data):", len(train_data))

    print("len(test_data):", len(test_data))

    id2word = corpora.Dictionary(train_data + test_data)

    train_texts = train_data
    test_texts = test_data
    train_corpus = [id2word.doc2bow(text) for text in train_texts]

    test_corpus = [id2word.doc2bow(text) for text in test_texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=50, random_state=100,
                                                update_every=1, chunksize=100, passes=10,
                                                alpha='auto', per_word_topics=True)

    from pprint import pprint

    print(lda_model.print_topics())

    # Compute Perplexity
    print('Perplexity: ', lda_model.log_perplexity(train_corpus))  # 越低越好

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=train_data + test_data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)  # 越高越好

    import pyLDAvis.gensim_models


    # vis = pyLDAvis.gensim_models.prepare(lda_model, train_corpus, id2word)

    # pyLDAvis.save_html(vis, 'lda_visualization1.html')

    train_cla = []
    test_cla = []
    for i, item in enumerate(test_corpus):
        topic = lda_model.get_document_topics(item)
        init = np.zeros(1000)
        for i, v in topic:
            init[i] = v
        test_cla.append(init)
        #print('第', i+1, '条记录分类结果:', topic)


    for i, item in enumerate(train_corpus):
        topic = lda_model.get_document_topics(item)
        init = np.zeros(1000)
        for i, v in topic:
            init[i] = v
        train_cla.append(init)

    from sklearn.svm import SVC

    clf = SVC(C=0.4, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto',
              kernel='poly', max_iter=- 1, probability=False, random_state=None, shrinking=True, tol=0.0000000001,
              verbose=False)
    clf = SVC()
    # 可以根据前面介绍的参数，做出相应改变观察结果变化
    y = []
    x = train_cla
    t = test_cla
    truth = []
    for i in range(1000):
        if i < 900:
            y.append(i // 180)
        else:
            truth.append((i - 900) // 20)
    clf.fit(x, y)
    print(y)
    pred = clf.predict(t).tolist()
    print(pred)
    print(truth)
    acc = 0.0
    for i in range(100):
        if pred[i] == truth[i]:
            acc = acc + 1
    acc = acc / 100
    print(acc)


    # vis = pyLDAvis.gensim_models.prepare(lda_model, test_corpus, id2word)

    # pyLDAvis.save_html(vis, 'lda_visualization2.html')
