# -*- coding:utf-8 -*-
import numpy as np

np.random.seed(0)
# 婵�娲诲嚱鏁�
def logistic(x):
    return 1/(1+np.exp(-x))

# 鍒涘缓涓�涓煩闃�
def random_function(m, n, fun=0.0):
    mat = []
    for i in range(m):
        mat.append([fun]*n)
    return mat

# 闅忔満浜х敓涓�浜涙暟
def rand(a, b):
    return (b-a) * np.random.random() + a
# 涓�涓缁忕綉缁滅被
class NN:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.hidden_weights = []
        self.hidden_bis = []
        self.output_bis = []

    # 鍒濆鍖栫綉缁�
    def setup(self, ni, nh, no):
        self.input_n = ni+1
        self.hidden_n = nh
        self.output_n = no
        self.input_cells = [1.0]*self.input_n
        self.hidden_cells = [1.0]*self.hidden_n
        self.output_cells = [1.0]*self.output_n
        self.input_weights = random_function(self.hidden_n, self.input_n)
        self.hidden_weights = random_function(self.output_n, self.hidden_n)
        self.hidden_bis = random_function(self.hidden_n, 1)
        self.output_bis = random_function(self.output_n, 1)

        for h in range(self.hidden_n):
            for i in range(self.input_n):
                self.input_weights[h][i] = rand(-1, 1)
        for o in range(self.output_n):
            for h in range(self.hidden_n):
                self.hidden_weights[o][h] = rand(-1, 1)
        for h in range(self.hidden_n):
            self.hidden_bis[h] = rand(-1, 1)
        for o in range(self.output_n):
            self.output_bis[o] = rand(-1, 1)

    # 姝ｅ悜璁＄畻
    def BPupward(self, inputs):
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[j][i]
            total += self.hidden_bis[j]
            self.hidden_cells[j] = logistic(total)

        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.hidden_weights[k][j]
            total += self.output_bis[k]
            self.output_cells = logistic(total)
        return self.output_cells

    # 鍙嶅悜鏇存柊鏉冨�煎拰鍋忓悜
    def back_update(self, case, label, learn):
        labelCla = self.BPupward(case)
        output_error = [0.0]*self.output_n
        hidden_error = [0.0]*self.hidden_n
        for o in range(self.output_n):
            output_error[o] = labelCla*(1 - labelCla)*(label - labelCla)
        for o in range(self.output_n):
            for h in range(self.hidden_n):
                hidden_error[h] = self.hidden_cells[h]*(1 - self.hidden_cells[h])*output_error[o]*self.hidden_weights[o][h]

        # 鏉冨�兼洿鏂�
        for o in range(self.output_n):
            for h in range(self.hidden_n):
                self.hidden_weights[o][h] += learn*output_error[o]*self.hidden_cells[h]

        for h in range(self.hidden_n):
            for i in range(self.input_n):
                self.input_weights[h][i] += learn*hidden_error[h]*self.input_cells[i]

        # 鍋忛噸鏇存柊
        for o in range(self.output_n):
            self.output_bis[o] += learn*output_error[o]

        for h in range(self.hidden_n):
            self.hidden_bis[h] += learn*hidden_error[h]

        return self.output_bis


    def test(self):
        datas = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
        labels = [[0], [1], [0], [0]]
        self.setup(2, 6, 1)
        learn = 0.05
        for l in range(10000):
            for d in range(len(datas)):
                self.back_update(datas[d], labels[d], learn=learn)

        # for x in range(len(datas)):
        #     print(self.BPupward(datas[x]))
        print(self.BPupward([0, 1]))
        print(self.BPupward([1, 1]))

if __name__=='__main__':
    nn = NN()
    nn.test()