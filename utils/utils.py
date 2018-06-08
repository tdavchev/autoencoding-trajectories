import numpy as np

from random import shuffle

class LoadTrajData(object):
    def __init__(self, contents='directions', tag='single', file_path='./data/timeseries_25_May_2018_18_51_49-walk3.npy'):
        self.char2Num = {}
        self.seqlen = {'inputs':[], 'targets':[]}
        self.max_len = {'inputs':0, 'targets':0}
        self.contents = contents
        self.tag = tag
        self.input_data, self.target_data = self.loadData(file_path)
    
    def toStringLocations(self, data):
        input_data = []
        target_data = []
        br = 0
        for entry in np.array(data)[:, 0]:
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

    def toSlopeWithDirections(self, data):
        input_data = []
        target_data = []
        br = 0
        for entry, label in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
            input_data.append([])
            target_data.append("")
            for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                input_data[br].append(x-y)
                if count <= self.points_of_split[br]:
                    target_data[br] += label[1] + " "
                else:
                    target_data[br] += label[2] + " "
                    
            target_data[br] = target_data[br][:-1] # account for empty space in the end..

            self.seqlen['inputs'].append(len(input_data[br]))
            # the +1 accounts for the <GO> symbol
            self.seqlen['targets'].append(len(target_data[br].split()) + 1)

            if self.max_len['inputs'] <= len(input_data[br]):
                self.max_len['inputs'] = len(input_data[br])

            if self.max_len['targets'] <= len(target_data[br].split()):
                self.max_len['targets'] = len(target_data[br].split())

            br += 1

        return input_data, target_data

    def toDwithDirections(self, data):
        input_data = []
        target_data = []
        br = 0
        for entry, label in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
            input_data.append([])
            target_data.append("")
            for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                input_data[br].append([x, y])
                if count <= self.points_of_split[br]:
                    target_data[br] += label[1] + " "
                else:
                    target_data[br] += label[2] + " "
                    
            target_data[br] = target_data[br][:-1] # account for empty space in the end..

            self.seqlen['inputs'].append(len(input_data[br]))
            # the +1 accounts for the <GO> symbol
            self.seqlen['targets'].append(len(target_data[br].split()) + 1)

            if self.max_len['inputs'] <= len(input_data[br]):
                self.max_len['inputs'] = len(input_data[br])

            if self.max_len['targets'] <= len(target_data[br].split()):
                self.max_len['targets'] = len(target_data[br].split())

            br += 1

        return input_data, target_data

    def toDirections(self, data):
        input_data = []
        target_data = []
        br = 0
        for entry, label in zip(np.array(data)[:, 0], np.array(data)[:, 1]):
            input_data.append("")
            target_data.append("")
            if self.tag == 'every': # use labels for every timestep
                for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                    # input_data[br] += str(x)+","+str(y) + " "
                    input_data[br] += str(x) + str(y) + " "
                    if count <= self.points_of_split[br]:
                        target_data[br] += label[1] + " "
                    else:
                        target_data[br] += label[2] + " "
                    
                target_data[br] = target_data[br][:-1] # account for empty space in the end..
            elif self.tag == 'single':
                for count, (x, y) in enumerate(zip(entry[0], entry[1])):
                    input_data[br] += str(x) + str(y) + " "

                target_data[br] = label[1] + " " + label[2]

            self.seqlen['inputs'].append(len(input_data[br]))
            # the +1 accounts for the <GO> symbol
            self.seqlen['targets'].append(len(target_data[br].split()) + 1)

            if self.max_len['inputs'] <= len(input_data[br]):
                self.max_len['inputs'] = len(input_data[br])

            if self.max_len['targets'] <= len(target_data[br].split()):
                self.max_len['targets'] = len(target_data[br].split())

            br += 1

        return input_data, target_data

    def augment(self, data):
        temp = []
        for x, y in data:
            if y[1] == 'right':
                if y[2] == 'left':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "left", "right", "eos"]))
                if y[2] == 'up':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "up" "right", "eos"]))
                if y[2] == 'down':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "down", "right", "eos"]))
            if y[1] == 'left':
                if y[2] == 'right':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "right", "left", "eos"]))
                if y[2] == 'up':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "up", "left", "eos"]))
                if y[2] == 'down':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "down", "left", "eos"]))
            if y[1] == 'up':
                if y[2] == 'right':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "right", "up", "eos"]))
                if y[2] == 'left':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "left" "up", "eos"]))
                if y[2] == 'down':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "down", "up", "eos"]))
            if y[1] == 'down':
                if y[2] == 'right':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "right", "down", "eos"]))
                if y[2] == 'up':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "up" "down", "eos"]))
                if y[2] == 'left':
                    temp.append(([x[0][::-1], x[1][::-1]], ["sos", "left", "down", "eos"]))

        for x, y in temp:
            data.append((x, y))

        return data

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
        if dataType == 'inputs':
            char2num = self._charToNum(data_points, 'locations')
        else:
            char2num = self._charToNum(data_points, contents)

        return char2num

    def convertData(self, data_points, _char2num, dataType, contents, pad=True):
        if dataType == 'targets' and contents == 'directions':
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

    def embedAsAsPoint(self, data_points, dataType='inputs'):
        pt = [list(point) + [[0,0]]*(self.max_len[dataType] - len(point)) for point in data_points]
        return np.array(pt)

    def embedData(self, data_points, contents='locations', padas='text', dataType='inputs', test=False, pad=True):
        # if target is point then needs implementing
        if padas == 'text':
            pt = self.embedAsString(data_points, contents, dataType, test, pad)
        elif padas == 'points':
            pt = self.embedAsAsPoint(data_points)

        return pt

    def _charToNum(self, data_points, contents='locations'):
        if contents == 'locations':
            u_characters = set(' '.join(data_points))
        elif contents == 'directions':
            u_characters = set(["left", "right", "up", "down"])
        return dict(zip(u_characters, range(1, len(u_characters) + 1))) # the <PAD> is 0

    def numToChar(self, _char2num):
        return dict(zip(_char2num.values(), _char2num.keys()))

    def splitData(self, raw_x, raw_y):
        complex_data = [
            ([         list(raw_x[:420]) + list(raw_x[420:622]),                       list(raw_y[:420]) + list(raw_y[420:622])],        ["sos","right", "down", "eos"]),
            ([    list(raw_x[4658:4835]) + list(raw_x[4835:4898]),                list(raw_y[4658:4835]) + list(raw_y[4835:4898])],      ["sos","right", "down", "eos"]),
            ([    list(raw_x[5150:5326]) + list(raw_x[5326:5404]),                list(raw_y[5150:5326]) + list(raw_y[5326:5404])],      ["sos","right", "down", "eos"]),
            ([    list(raw_x[:420]-0.03) + list(raw_x[420:542]-0.03),              list(raw_y[:420]-0.2) + list(raw_y[420:542]-0.2)],    ["sos","right", "down", "eos"]),
            ([      list(raw_x[270:420]) + list(raw_x[420:522]),                    list(raw_y[270:420]) + list(raw_y[420:522])],        ["sos","right", "down", "eos"]),
            ([    list(raw_x[2330:2563]) + list(raw_x[2563:2742]),                list(raw_y[2330:2563]) + list(raw_y[2563:2742])],      ["sos","right", "down", "eos"]),
            ([    list(raw_x[2430:2563]) + list(raw_x[2563:2622]),                list(raw_y[2430:2563]) + list(raw_y[2563:2622])],      ["sos","right", "down", "eos"]),
            ([    list(raw_x[3575:3842]) + list(raw_x[3842:4062]),                list(raw_y[3575:3842]) + list(raw_y[3842:4062])],      ["sos","right", "down", "eos"]),
            ([    list(raw_x[3675:3842]) + list(raw_x[3842:3950]),                list(raw_y[3675:3842]) + list(raw_y[3842:3950])],      ["sos","right", "down", "eos"]),
            ([      list(raw_x[362:636]) + list(raw_x[636:858]),                    list(raw_y[362:636]) + list(raw_y[636:858])],        ["sos","down", "left", "eos"]),
            ([      list(raw_x[462:636]) + list(raw_x[636:758]),                    list(raw_y[462:636]) + list(raw_y[636:758])],        ["sos","down", "left", "eos"]),
            ([    list(raw_x[1295:1490]) + list(raw_x[1490:1858]),                list(raw_y[1295:1490]) + list(raw_y[1490:1858])],      ["sos","down", "left", "eos"]),
            ([    list(raw_x[1345:1490]) + list(raw_x[1490:1658]),                list(raw_y[1345:1490]) + list(raw_y[1490:1658])],      ["sos","down", "left", "eos"]),
            ([list(raw_x[1345:1490]-0.3) + list(raw_x[1490:1658]-0.3),            list(raw_y[1345:1490]) + list(raw_y[1490:1658])],      ["sos","down", "left", "eos"]),
            ([list(raw_x[1345:1490]-0.3) + list(raw_x[1490:1658]-0.3),       list(raw_y[1345:1490]+0.25) + list(raw_y[1490:1658]+0.25)], ["sos","down", "left", "eos"]),
            ([    list(raw_x[2537:2741]) + list(raw_x[2741:3050]),                list(raw_y[2537:2741]) + list(raw_y[2741:3050])],      ["sos","down", "left", "eos"]),
            ([    list(raw_x[2617:2741]) + list(raw_x[2741:3050]),                list(raw_y[2617:2741]) + list(raw_y[2741:3050])],      ["sos","down", "left", "eos"]),
            ([    list(raw_x[2617:2741]) + list(raw_x[2741:2950]),                list(raw_y[2617:2741]) + list(raw_y[2741:2950])],      ["sos","down", "left", "eos"]),
            ([    list(raw_x[4062:4478]) + list(raw_x[4478:4618]),                list(raw_y[4062:4478]) + list(raw_y[4478:4618])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[4262:4478]) + list(raw_x[4478:4638]),                list(raw_y[4262:4478]) + list(raw_y[4478:4638])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[2742:3105]) + list(raw_x[3105:3575]),                list(raw_y[2742:3105]) + list(raw_y[3105:3575])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[2942:3105]) + list(raw_x[3105:3575]),                list(raw_y[2942:3105]) + list(raw_y[3105:3575])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[2942:3105]) + list(raw_x[3105:3375]),                list(raw_y[2942:3105]) + list(raw_y[3105:3375])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[2742:3105]) + list(raw_x[3105:3375]),                list(raw_y[2742:3105]) + list(raw_y[3105:3375])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[1500:1860]) + list(raw_x[1860:2130]),                list(raw_y[1500:1860]) + list(raw_y[1860:2130])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[1700:1860]) + list(raw_x[1860:2030]),                list(raw_y[1700:1860]) + list(raw_y[1860:2030])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[4062:4480]) + list(raw_x[4480:4638]),                list(raw_y[4062:4480]) + list(raw_y[4480:4638])],      ["sos","left", "up", "eos"]),
            ([    list(raw_x[1842:2155]) + list(raw_x[2155:2552]),                list(raw_y[1842:2155]) + list(raw_y[2155:2552])],      ["sos","up", "right", "eos"]),
            ([    list(raw_x[4535:4656]) + list(raw_x[4656:4732]),                list(raw_y[4535:4656]) + list(raw_x[4656:4732])],      ["sos","up", "right", "eos"])
        ]
        # not sure about the fix/hack revisit!
        return np.array(complex_data).tolist() # fixes some of the typing differences

    def getSplitPoints(self):
        dicto = [
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
        ]
        reversed_dicto = [
            622-420,   #  0
            4898-4835, #  1
            5404-5326, #  2
            542-420,   #  3
            522-270,   #  4
            2742-2563, #  5
            2622-2563, #  6
            4062-3842, #  7
            3950-3842, #  8
            858-636,   #  9
            758-636,   # 10
            1858-1490, # 11
            1658-1490, # 12
            1658-1490, # 13
            1658-1490, # 14
            3050-2741, # 15
            3050-2741, # 16
            2950-2741, # 17
            4618-4478, # 18
            4638-4478, # 19
            3575-3105, # 20
            3575-3105, # 21
            3575-3105, # 22
            3575-3105, # 23
            2130-1860, # 24
            2030-1860, # 25
            4638-4480, # 26
            2552-2155, # 27
            4732-4656  # 28
        ]

        dicto = np.concatenate((dicto, reversed_dicto))
        return dicto

    def loadData(self, file_path):
        raw_data = np.load(file_path)
        raw_x, raw_y = raw_data[:, 0], raw_data[:, 1]
        raw_y = np.abs(raw_y - np.max(raw_y)) # flip it
        self.points_of_split = self.getSplitPoints()
        split_data = self.splitData(raw_x, raw_y)
        data = self.augment(split_data)
        if self.contents == 'locations':
            input_data, target_data = self.toStringLocations(data)
        elif self.contents == 'directions':
            input_data, target_data = self.toDirections(data)
        elif self.contents == '2D-directions':
            input_data, target_data = self.toDwithDirections(data)

        return input_data, target_data

    def batch_data(self, data, labels, seqlen_idx, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        shuffle = np.random.permutation(len(data))

        data = data[shuffle]
        labels = labels[shuffle]
        seqlen_idx = [seqlen_idx[i] for i in shuffle]
        seqlen = [self.seqlen['inputs'][i] for i in seqlen_idx]
        y_seqlen = [self.seqlen['targets'][i] for i in seqlen_idx]
        p_of_split = [self.points_of_split[i] for i in seqlen_idx]
        start = 0

        while start + batch_size <= len(data):
            yield data[start:start+batch_size], labels[start:start+batch_size], seqlen[start:start+batch_size], y_seqlen[start:start+batch_size], p_of_split[start:start+batch_size], shuffle[start:start+batch_size]
            start += batch_size
