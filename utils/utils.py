import numpy as np

from random import shuffle

class LoadTrajData(object):
    def __init__(self, contents='2D-directions', tag='single', input_type='centered_at_start', target_type='normalized_time', file_path='./data/data.txt', file_complex_path='./data/one_turn_data.txt', raw_file_path='./data/timeseries_25_May_2018_18_51_49-walk3.npy'):
        self.char2Num = {}
        self.seqlen = {'inputs':[], 'targets':[], 'actions':[]}
        self.max_len = {'inputs':0, 'targets':0, 'actions':0}
        self.contents = contents
        self.tag = tag
        self.input_type = input_type
        self.target_type = target_type
        self.loadRawData(raw_file_path)
        self.data_dictionary = {"indexes":[], "alphas":[], "targets":[]}
        data = self.preprocess(file_path=file_path, file_complex_path=file_complex_path)
        self.input_data, self.action_data, self.target_data = self.assignData(data)
    
    def toStringLocations(self, data):
        input_data = []
        target_data = []
        br = 0
        for entry in np.array(data):
            input_data.append("")
            target_data.append("")
            for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                input_data[br] += str(x)+","+str(y) + " "
                if count <= self.points_of_split[br]+3 and count >= self.points_of_split[br]-3:
                    for _ in range(10):
                        target_data[br] += 'B'
                else:
                    target_data[br] += str(x)+","+str(y) + " "

            self.seqlen['inputs'].append(len(input_data[br]))
            # the +1 accounts for the <GO> symbol
            self.seqlen['targets'].append(len(target_data[br]) + 1)
            
            if self.max_len['inputs'] <= len(input_data[br]):
                self.max_len['inputs'] = len(input_data[br])

            if self.max_len['targets'] <= len(target_data[br]):
                self.max_len['targets'] = len(target_data[br])

            br += 1

        return input_data, target_data

    def directions(self, br):
        string_data = ""
        for i in range(len(self.data_dictionary["targets"][br])):
            string_data += self.data_dictionary["targets"][br][i]
            if i < len(self.data_dictionary["targets"][br])-1:
                string_data += " "

        return string_data

    def toDwithDirections(self, data):
        input_data = []
        target_data = []
        actions_data = []
        br = 0
        for entry, label in zip(data, self.data_dictionary["targets"]):
            input_data.append([])
            actions_data.append("")
            target_data.append(0.0)
            if self.tag == 'every': # use labels for every timestep
                for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                    if self.input_type == 'basic':
                        input_data[br].append([x, y])
                    elif self.input_type == 'centered_at_start':
                        input_data[br].append([x - entry[0][0], y - entry[1][0]])

                    addLabel = False
                    for i, pos in enumerate(self.points_of_split[br]):
                        if count <= pos and not addLabel:
                            addLabel = True
                            actions_data[br] += label[i] + " "
                        if i == len(self.points_of_split[br])-1 and not addLabel:
                            addLabel = True
                            actions_data[br] += label[i+1]

            elif self.tag == 'single':
                for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                    if self.input_type == 'basic':
                        input_data[br].append([x, y])
                    elif self.input_type == 'centered_at_start':
                        input_data[br].append([x - entry[0][0], y - entry[1][0]])

                actions_data[br] = self.directions(br)

            if self.target_type == 'normalized_time':
                target_data[br] = self.points_of_split[br]/len(input_data[br])
                self.max_len['targets'] = 1
                self.seqlen['targets'].append(1)
            elif self.target_type == 'basic':
                if self.input_type == 'basic':
                    target_data[br].append([entry[0][self.points_of_split[br]], entry[1][self.points_of_split[br]]])
                elif self.input_type == 'centered_at_start':
                    target_data[br].append([entry[0][self.points_of_split[br]] - entry[0][0], entry[1][self.points_of_split[br]] - entry[1][0]])
                    target_data[br].append([entry[0][self.points_of_split[br]] - entry[0][0], entry[1][self.points_of_split[br]] - entry[1][0]])

                if self.max_len['targets'] <= len(target_data[br]):
                    self.max_len['targets'] = len(target_data[br])

                self.seqlen['targets'].append(len(target_data[br]))

            self.seqlen['inputs'].append(len(input_data[br]))
            # the +1 accounts for the <GO> symbol
            self.seqlen['actions'].append(len(actions_data[br].split()) + 1)

            if self.max_len['inputs'] <= len(input_data[br]):
                self.max_len['inputs'] = len(input_data[br])

            if self.max_len['actions'] <= len(actions_data[br].split()):
                self.max_len['actions'] = len(actions_data[br].split())

            br += 1

        return input_data, actions_data, target_data

    def toDirections(self, data):
        input_data = []
        target_data = []
        br = 0
        for entry, label in zip(np.array(data), self.data_dictionary["targets"]):
            input_data.append("")
            target_data.append("")
            if self.tag == 'every': # use labels for every timestep
                for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                    # input_data[br] += str(x)+","+str(y) + " "
                    input_data[br] += str(x) + str(y) + " "
                    addLabel = False
                    for i, pos in enumerate(self.points_of_split[br]):
                        if count <= pos and not addLabel:
                            addLabel = True
                            target_data[br] += label[i] + " "
                        if i == len(self.points_of_split[br])-1 and not addLabel:
                            addLabel = True
                            target_data[br] += label[i+1]

                target_data[br] = target_data[br][:-1] # account for empty space in the end..
            elif self.tag == 'single':
                for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                    input_data[br] += str(x) + str(y) + " "

                target_data[br] = self.directions(br)

            self.seqlen['inputs'].append(len(input_data[br]))
            # the +1 accounts for the <GO> symbol
            self.seqlen['targets'].append(len(target_data[br].split()) + 1)

            if self.max_len['inputs'] <= len(input_data[br]):
                self.max_len['inputs'] = len(input_data[br])

            if self.max_len['targets'] <= len(target_data[br].split()):
                self.max_len['targets'] = len(target_data[br].split())

            br += 1

        return input_data, target_data

    def update(self, indexes, targets,  alphas=[0,0]):
        self.data_dictionary["indexes"].append(indexes)
        self.data_dictionary["alphas"].append(alphas)
        self.data_dictionary["targets"].append(targets)

    def augment(self, data):
        reversed_data = []
        for br, (values, x) in enumerate(zip(data, np.squeeze(self.data_dictionary["indexes"]))):
            reversed_data.append([values[0][::-1], values[1][::-1]])
            if self.data_dictionary["targets"][br][0] == 'right':
                if len(self.data_dictionary["targets"][br]) > 1:
                    if self.data_dictionary["targets"][br][1] == 'left':
                        self.update([x[0], x[1]], ["right", "left"])
                    if self.data_dictionary["targets"][br][1] == 'up':
                        self.update([x[0], x[1]], ["down", "left"])
                    if self.data_dictionary["targets"][br][1] == 'down':
                        self.update([x[0], x[1]], ["up", "left"])
                else:
                    self.update([x[0], x[1]], ["left"])

            if self.data_dictionary["targets"][br][0] == 'left':
                if len(self.data_dictionary["targets"][br]) > 1:
                    if self.data_dictionary["targets"][br][1] == 'right':
                        self.update([x[0], x[1]] ["left", "right"])
                    if self.data_dictionary["targets"][br][1] == 'up':
                        self.update([x[0], x[1]], ["down", "right"])
                    if self.data_dictionary["targets"][br][1] == 'down':
                        self.update([x[0], x[1]], ["up", "right"])
                else:
                        self.update([x[0], x[1]], ["right"])

            if self.data_dictionary["targets"][br][0] == 'up':
                if len(self.data_dictionary["targets"][br]) > 1:
                    if self.data_dictionary["targets"][br][1] == 'right':
                        self.update([x[0], x[1]], ["left", "down"])
                    if self.data_dictionary["targets"][br][1] == 'left':
                        self.update([x[0], x[1]], ["right", "down"])
                    if self.data_dictionary["targets"][br][1] == 'down':
                        self.update([x[0], x[1]], ["up", "down"])
                else:
                        self.update([x[0], x[1]], ["down"])

            if self.data_dictionary["targets"][br][0] == 'down':
                if len(self.data_dictionary["targets"][br]) > 1:
                    if self.data_dictionary["targets"][br][1] == 'right':
                        self.update([x[0], x[1]], ["left", "up"])
                    if self.data_dictionary["targets"][br][1] == 'up':
                        self.update([x[0], x[1]], ["down", "up"])
                    if self.data_dictionary["targets"][br][1] == 'left':
                        self.update([x[0], x[1]], ["right", "up"])
                else:
                        self.update([x[0], x[1]], ["up"])

        return np.concatenate((data, reversed_data))

    def calculateAccuracy(self, tar_v, pred_v, batch_y_seqlen, acc=[]):
        for idx, (val, pred_val) in enumerate(zip(tar_v, pred_v)):
            predicted_length = len(pred_val)
            for pos, (v_pt, p_pt) in enumerate(zip(val[:predicted_length], pred_val)):
                if batch_y_seqlen[idx] >= pos:
                    if v_pt == p_pt:
                        acc.append(1)
                    else:
                        acc.append(0)
            # treat mispredicted end of sentence as a mistake
            for _ in range(len(val[predicted_length:])):
                acc.append(0)

        # this should compute per batch not over all.
        return np.mean(acc)

    def convertChar2Num(self, data_points, dataType, contents='locations'):
        if dataType == 'inputs' or dataType == 'actions':
            if contents == 'directions':
                contents = 'locations'

            char2num = self._charToNum(data_points, contents)
        else:
            char2num = self._charToNum(data_points, contents)

        return char2num

    def convertData(self, data_points, _char2num, dataType, contents, pad=True):
        if (dataType == 'targets' and contents == 'directions') or (dataType == 'actions' and contents == '2D-directions'):
            if pad:
                pt = [[_char2num[p_] for p_ in point.split()] + [_char2num['<END>']] + [_char2num['<PAD>']]*(self.max_len[dataType] - len(point.split())) for point in data_points]
            else:
                pt = [[_char2num[p_] for p_ in point.split()] for point in data_points]
        else:
            if pad:
                pt = [[_char2num[p_] for p_ in point] + [_char2num['<END>']] + [_char2num['<PAD>']]*(self.max_len[dataType] - len(point)) for point in data_points]
            else:
                pt = [[_char2num[p_] for p_ in point] for point in data_points]

        # moje da se naloji da go enforcenesh da e 0 (though that's a hack..)
        if dataType == 'targets':
            _char2num['<GO>'] = len(_char2num)
            pt = [[_char2num['<GO>']] + point for point in pt]

        return pt

    def embedAsString(self, data_points, contents, dataType='inputs', test=False, pad=True):
        # Not the pretiest method...reduce the if-else cases ..
        if pad:
            _char2num = {'<PAD>':0}
        else:
            _char2num = {}

        _char2num.update(self.convertChar2Num(data_points, dataType, contents))
        _char2num['<END>'] = len(_char2num)
        if not test: # assuming no new characters introduced in test.
            self.char2Num[dataType] = _char2num

        return np.array(self.convertData(data_points, _char2num, dataType, contents, pad))

    def embedAsAsPoint(self, data_points, contents, idxes=[], dataType='inputs', test=False, pad=True):
        pt = [list(point) + [[0,0]]*(self.max_len[dataType] - len(point)) for point in data_points]
        if contents == '2D-directions':
            acts = [self.action_data[i] for i in idxes] # data is shuffled after split
            a_pt = self.embedAsString(acts, contents, 'actions', test, pad)
        return [np.array(pt), np.array(a_pt)]

    def embedData(self, data_points, idxes=[], contents='locations', padas='text', dataType='inputs', test=False, pad=True):
        # if target is point then needs implementing
        if padas == 'text':
            pt = self.embedAsString(data_points, contents, dataType, test, pad)
        elif padas == 'numeric':
            pt = self.embedAsAsPoint(data_points, contents, idxes, test=test, pad=pad)

        return pt

    def _charToNum(self, data_points, contents='locations'):
        if contents == 'locations':
            u_characters = set(' '.join(data_points))
        elif contents == 'directions' or '2D-directions':
            u_characters = set(["left", "right", "up", "down"])
        return dict(zip(u_characters, range(1, len(u_characters) + 1))) # the <PAD> is 0

    def numToChar(self, _char2num):
        return dict(zip(_char2num.values(), _char2num.keys()))

    def loadData(self, data_path):
        br = len(self.data_dictionary["indexes"])
        # data = {"indexes":[], "alphas":[], "directions":[]}
        with open(data_path, 'r') as content:
            for entry in content.readlines():
                self.data_dictionary["indexes"].append([])
                self.data_dictionary["alphas"].append([])
                self.data_dictionary["targets"].append([])
                _input = entry.split("\n")[0]
                _input = [item.strip() for item in _input.split(",")]
                self.data_dictionary["indexes"][br].append([int(_input[0]), int(_input[1])])  
                self.data_dictionary["alphas"][br].append([float(_input[2]), float(_input[3])])
                for i in range(4,len(_input)):
                    self.data_dictionary["targets"][br].append(_input[i][1:-1])
                br += 1

    def splitData(self, data, data_indexes, alphas):
        for i, idxs in enumerate(data_indexes):
            data.append(
                [
                    self.raw_x[idxs[0]:idxs[1]] + alphas[i][0],
                    self.raw_y[idxs[0]:idxs[1]] + alphas[i][1]
                ]
            )

        # fixes some of the typing differences, don't think I need it anymore.
        return data

    def getSplitComplexPoints(self, split_data):
        dicto = []
        temp_dicto = []
        for i in range(82): # check that
            dicto.append(len(split_data[i][0]))
            temp_dicto.append(len(split_data[i][0]))

        for ele in [
            420,        #  0
            4835-4658,  #  1
            5326-5150,  #  2
            420,        #  3
            420-270,    #  4
            2563-2330,  #  5
            2563-2430,  #  6
            3842-3575,  #  7
            3842-3675,  #  8
            636-362,    #  9
            636-462,    # 10
            1490-1295,  # 11
            1490-1345,  # 12
            1490-1345,  # 13
            1490-1345,  # 14
            2741-2537,  # 15
            2741-2617,  # 16
            2741-2617,  # 17
            4478-4062,  # 18
            4478-4262,  # 19
            3105-2742,  # 20
            3105-2942,  # 21
            3105-2942,  # 22
            3105-2742,  # 23
            1860-1500,  # 24
            1860-1700,  # 25
            4480-4062,  # 26
            2155-1842,  # 27
            4656-4535   # 28
        ]:
            dicto.append(ele)
        reversed_dicto = [
            622-420,   #  0 29
            4898-4835, #  1 30
            5404-5326, #  2 31
            542-420,   #  3 32
            522-420,   #  4 33
            2742-2563, #  5 34
            2622-2563, #  6 35
            4062-3842, #  7 36
            3950-3842, #  8 37
            858-636,   #  9 38
            758-636,   # 10 39
            1858-1490, # 11 40
            1658-1490, # 12 41
            1658-1490, # 13 42
            1658-1490, # 14 43
            3050-2741, # 15 44
            3050-2741, # 16 45
            2950-2741, # 17 46
            4618-4478, # 18 47
            4638-4478, # 19 48
            3575-3105, # 20 49
            3575-3105, # 21 50
            3375-3105, # 22 51
            3375-3105, # 23 52
            2130-1860, # 24 53
            2030-1860, # 25 54
            4638-4480, # 26 55
            2552-2155, # 27 56
            4732-4656  # 28 57
        ]

        dicto = np.concatenate((dicto, temp_dicto))
        dicto = np.concatenate((dicto, reversed_dicto))
        return dicto


    def getSplitPoints(self, data_points, reversed_data_points):
        dicto = [len(ele)-1 for ele in data_points]
        reversed_dicto = [len(ele)-1 for ele in data_points]

        dicto = np.concatenate((dicto, reversed_dicto))
        return dicto

    def normalise(self, x, y):
        return [(x-np.mean(x))/np.std(x), (y-np.mean(y))/np.std(y)]

    def assignData(self, data):
        if self.contents == 'locations':
            input_data, target_data = self.toStringLocations(data)
            action_data = target_data
        elif self.contents == 'directions':
            input_data, target_data = self.toDirections(data)
            action_data = target_data
        elif self.contents == '2D-directions':
            input_data, action_data, target_data = self.toDwithDirections(data)

        return input_data, action_data, target_data

    def preprocess(self, file_path='', file_complex_path='', split_data=[]):
        if file_path != '':
            self.loadData(file_path)
        if file_complex_path != '':
            self.loadData(file_complex_path)

        split_data = self.splitData(split_data, np.squeeze(self.data_dictionary["indexes"]), np.squeeze(self.data_dictionary["alphas"]))
        split_data = self.augment(split_data)

        self.points_of_split = self.getSplitComplexPoints(split_data)

        return split_data

    def loadRawData(self, file_path):
        raw_data = np.load(file_path)
        self.raw_x, self.raw_y = raw_data[:, 0], raw_data[:, 1]
        self.raw_y = np.abs(self.raw_y - np.max(self.raw_y)) # flip it
        self.raw_x, self.raw_y = self.normalise(self.raw_x, self.raw_y)

    def batch_data(self, data, labels, seqlen_idx, batch_size, actions=[]):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        shuffle = np.random.permutation(len(data))

        data = data[shuffle]
        
        labels = [labels[i] for i in shuffle]

        seqlen_idx = [seqlen_idx[i] for i in shuffle]

        seqlen = [self.seqlen['inputs'][i] for i in seqlen_idx]
        y_seqlen = [self.seqlen['targets'][i] for i in seqlen_idx]


        if len(actions) > 0:
            actions = actions[shuffle]
            a_seqlen = [self.seqlen['actions'][i] for i in seqlen_idx]
        else:
            actions = np.zeros(len(shuffle))
            a_seqlen = np.zeros(len(seqlen_idx))

        p_of_split = [self.points_of_split[i] for i in seqlen_idx]
        start = 0

        while start + batch_size <= len(data):
            yield {
                "inputs": data[start:start+batch_size],
                "actions": actions[start:start+batch_size],
                "targets": labels[start:start+batch_size],
                "seqlen": seqlen[start:start+batch_size],
                "seqlen_a": a_seqlen[start:start+batch_size],
                "seqlen_y": y_seqlen[start:start+batch_size],
                "points_of_split": p_of_split[start:start+batch_size],
                "shuffle_ids": shuffle[start:start+batch_size]
                }
            start += batch_size
