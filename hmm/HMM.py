# coding=utf-8
# Reference:**********************************************
# @Time     : 2023/2/6 2:57 PM
# @Author   : huangyajian
# @File     : HMM.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************


class HMM():
    def __init__(self):
        # self.char_set = set()
        self.state_list = ['B', 'M', 'E', 'S']
        self.trans_dict = None
        self.emit_dict = None
        self.start_dict = None

    def init(self):
        trans_dict = {}  # 存储状态转移概率
        emit_dict = {}  # 发射概率(状态->词语的条件概率)
        count_dict = {}  # 存储所有状态序列 ，用于归一化分母
        start_dict = {}  # 存储状态的初始概率

        for from_state in self.state_list:
            trans_dict[from_state] = {}
            for to_state in self.state_list:
                trans_dict[from_state][to_state] = 0.0
        for state in self.state_list:
            start_dict[state] = 0.0
            emit_dict[state] = {}
            count_dict[state] = 0
        # print(trans_dict)  # {'B': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}, 'M': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}, 'E': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}, 'S': {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}}
        # print(emit_dict) # {'B': {}, 'M': {}, 'E': {}, 'S': {}}
        # print(start_dict) # {'B': 0.0, 'M': 0.0, 'E': 0.0, 'S': 0.0}
        # print(count_dict) #{'B': 0, 'M': 0, 'E': 0, 'S': 0}
        return trans_dict, emit_dict, start_dict, count_dict

    def get_word_status(self, word):
        word_status = []
        if len(word) == 1:
            word_status.append('S')
        elif len(word) == 2:
            word_status = ['B', 'E']
        else:
            M_num = len(word) - 2
            M_list = ['M'] * M_num
            word_status.append('B')
            word_status.extend(M_list)
            word_status.append('E')
        return word_status

    def train(self, train_filepath):
        """
        这里通过统计方法直接计算了pi、A、B，没有使用em算法
        """
        trans_dict, emit_dict, start_dict, count_dict = self.init()
        f = open(train_filepath, 'r')
        lines = f.readlines()
        line_cnt = len(lines)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            char_list = []
            for i in range(len(line)):
                if line[i] == ' ':
                    continue
                char_list.append(line[i])

            # self.char_set = set(char_list)
            word_list = line.split(" ")
            line_status = []

            for word in word_list:
                line_status.extend(self.get_word_status(word))

            if len(char_list) != len(line_status):
                continue

            for i in range(len(line_status)):
                if i == 0:
                    start_dict[line_status[0]] += 1
                    count_dict[line_status[0]] += 1
                else:
                    trans_dict[line_status[i - 1]][line_status[i]] += 1
                    count_dict[line_status[i]] += 1
                    if char_list[i] not in emit_dict[line_status[i]]:
                        emit_dict[line_status[i]][char_list[i]] = 0.0
                    else:
                        emit_dict[line_status[i]][char_list[i]] += 1

        # 归一化
        for key in start_dict:
            start_dict[key] = start_dict[key] * 1.0 / line_cnt

        for from_key in trans_dict:
            for to_key in trans_dict[from_key]:
                trans_dict[from_key][to_key] = trans_dict[from_key][to_key] / count_dict[from_key]

        for key in emit_dict:
            for word in emit_dict[key]:
                emit_dict[key][word] = emit_dict[key][word] / count_dict[key]

        self.trans_dict = trans_dict
        self.start_dict = start_dict
        self.emit_dict = emit_dict

    def viterbi(self, obs):
        v = [{}]
        path = {}
        for y in self.state_list:
            v[0][y] = self.start_dict[y] * self.emit_dict[y].get(obs[0], 0)  # 在位置0，以y状态为末尾的状态序列的最大概率
            path[y] = [y]

        for t in range(1, len(obs)):
            v.append({})
            new_path = {}
            for y in self.state_list:
                state_path = (
                    [(v[t - 1][y0] * self.trans_dict[y0].get(y, 0) * self.emit_dict[y].get(obs[t], 0), y0) for y0 in
                     self.state_list if v[t - 1][y0] > 0])  # 计算前一个隐状态y0转移到当前隐状态y且观测状态为obs[t]的概率
                if state_path == []:
                    (prob, state) = (0.0, 'S')
                else:
                    (prob, state) = max(state_path)  # 维特比算法的精髓，选择到当前状态最大的那条路径
                v[t][y] = prob
                new_path[y] = path[state] + [y]  # 记录历史路径
            path = new_path
        (prob, state) = max([(v[len(obs) - 1][y], y) for y in self.state_list])  # 选择概率最大的最后一个状态，然后反回前面的路径
        return (prob, path[state])

    def cut(self, sent):
        prob, pos_list = self.viterbi(sent)
        seglist = list()
        word = list()
        for index in range(len(pos_list)):
            if pos_list[index] == 'S':
                word.append(sent[index])
                seglist.append(word)
                word = []
            elif pos_list[index] in ['B', 'M']:
                word.append(sent[index])
            elif pos_list[index] == 'E':
                word.append(sent[index])
                seglist.append(word)
                word = []
        seglist = [''.join(tmp) for tmp in seglist]

        return seglist


def main():
    hmm = HMM()
    hmm.train('./train.txt')

    sent = '我们在野生动物园玩'
    print(hmm.cut(sent))


if __name__ == '__main__':
    main()
