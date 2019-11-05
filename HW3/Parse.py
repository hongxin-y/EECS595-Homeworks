import os
import sys
sys.setrecursionlimit(1000000)

def gram2file(filename):
    for line in open(filename):
        build_map(line, gram, vocab)
    #Write the grammar into file GrammarFile.txt.
    with open("GrammarFile.txt", 'w') as f:
        for idx in gram:
            for pair in gram[idx]:
                context = idx + " -> "
                for ele in pair:
                    context += ele + " "
                context += ": " + str(gram[idx][pair]) + "\n"
                f.write(context)
        f.write("\n")
        for idx in vocab:
            for word in vocab[idx]:
                context = idx + " -> "
                context += word + " "
                context += ": " + str(vocab[idx][word]) + "\n"
                f.write(context)

def build_CFG(filename):
    gram = {}
    vocab = {}
    if os.path.exists(filename):
        isGram = True
        for line in open(filename):
            if line == "\n":
                isGram = False
                continue
            words = line.split()
            idx = words[0]
            i = 2
            if isGram:
                p = []
                while(words[i] != ':'):
                    p.append(words[i])
                    i += 1
                pair = tuple(p)
                freq = int(words[i+1])
                if idx in gram:
                    gram[idx][pair] = freq
                else:
                    gram[idx] = {pair: freq}
            else:
                word = words[i]
                freq = int(words[4])
                if idx in vocab:
                    vocab[idx][word] = freq
                else:
                    vocab[idx] = {word: freq}
    gram = CNF(gram)
    for k in gram.keys():
        cnt = sum(gram[k].values())
        for kk in gram[k].keys():
            gram[k][kk] /= cnt
    for k in vocab.keys():
        cnt = sum(vocab[k].values())
        for kk in vocab[k].keys():
            vocab[k][kk] /= cnt    
    return gram, vocab

def build_map(line, gram, vocab):
    words = line.split()
    if len(words) <= 2:
        idx = words[0][1:]
        w = words[1][:-1]
        if idx in vocab:
            if w in vocab[idx]:
                vocab[idx][w] += 1
            else:
                vocab[idx][w] = 1
        else:
            vocab[idx] = {w:1}
        return 
    leftCnt = -1
    rightCnt = 0
    pair = []
    idx = words[0][1:]
    left = 0
    right = 0
    for i in range(len(line)):
        if line[i] == '[':
            if leftCnt == 0:
                left = i
            leftCnt += 1
        elif line[i] == ']':
            rightCnt += 1
            if leftCnt == rightCnt:
                right = i
                build_map(line[left: right+1], gram, vocab)
                pair.append(line[left: right+1].split()[0][1:])
                leftCnt = 0
                rightCnt = 0
    p = tuple(pair)
    if idx in gram:
        if p in gram[idx]:
            gram[idx][p] += 1
        else:
            gram[idx][p] = 1
    else:
        gram[idx] = {p:1}

def CNF(gram):
    tmp = 'X'
    idx = 0
    for k in list(gram.keys()):
        for p in list(gram[k]):
            pair = p
            while len(pair) > 2:
                X = tmp + str(idx)
                idx += 1
                subpair = pair[:2]
                gram[X] = {subpair:gram[k][pair]}
                new_pair = (X,) +  pair[2:]
                gram[k][new_pair] = gram[k][pair]
                del gram[k][pair]
                pair = new_pair
    return gram
                

def parse(parse_filename, test_filename, output_filename, gram, vocab):
    res = []
    for line in open(parse_filename, 'r'):
        words = line.split()
        if words == []:
            break
        n = len(words)
        dp = [[{} for _ in range(n)] for __ in range(n)]
        paths = [[{} for _ in range(n)] for __ in range(n)]
        for i in range(n):
            for k in vocab.keys():
                if words[i] in vocab[k]:
                    dp[i][i][k] = vocab[k][words[i]]
        
        for j in range(n):
            for i in range(j-1,-1,-1):
                label = None
                for key in gram.keys():
                    maximal = 0
                    for k in range(0,j-i):
                        for l1 in dp[i][i+k]:
                            for l2 in dp[i+k+1][j]:
                                pair = (l1, l2)
                                if pair in gram[key]:
                                    prob = dp[i][i+k][l1]*dp[i+k+1][j][l2]*gram[key][pair]
                                    dp[i][j][key] = prob
                                    if prob >= maximal:
                                        maximal = prob
                                        paths[i][j][key] = (k, l1, l2)
                    if maximal != 0:
                        dp[i][j][key] = maximal

        p = backtrace(n, paths, dp, 0, n-1, 'S', words, vocab)
        res.append(" ".join(p.split()))
    
    acc = None
    if os.path.exists(test_filename):
        i = 0
        acc = 0
        for line in open(test_filename, 'r'):
            if line == "" or i >= len(res):
                break;
            t = " ".join(line[:-1].split())
            if t == res[i] :
                acc += 1
            i += 1
        acc = acc/i

    with open(output_filename, 'w') as f:
        for r in res:
            f.write(r + "\n")
    return acc

def backtrace(n, paths, dp, i, j, label, words, vocab):
    l = label
    if l not in paths[i][j]:
        return "[" + l + " " + words[i] + "]" if l in vocab and words[i] in vocab[l] else "FAIL"

    k, l1, l2 = paths[i][j][l]
    left = backtrace(n, paths, dp, i, i+k, l1, words, vocab)
    right = backtrace(n, paths, dp, i+k+1, j, l2, words, vocab)
    
    if left == "FAIL" or right == "FAIL":
        return "FAIL"
    path = "[" + l + " " + left + " " + right + "]" if l[0] != 'X' else left + " " + right  
    return path

if __name__ == "__main__":
    if len(sys.argv) == 1:
        outputFile = "Output.txt"
        inputFile = "TestingRaw.txt"
        GrammarFile = "GrammarFile.txt"
    elif len(sys.argv) != 4:
        print("Please type correct number of parameters.")
        sys.exit()
    else:
        inputFile, GrammarFile, outputFile = sys.argv[1:]
    gram, vocab = build_CFG(GrammarFile)
    acc = parse(inputFile, "TestingTree.txt", outputFile, gram, vocab)
    print("Accuracy is {:.2f}% on test data.".format(acc*100) if acc != None else "There is no text file.")