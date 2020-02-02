import os
from data import dataset
from nnet.network import make_net

graph_types = ['tree', '2p']
var_annot_lim = [1, 9]
read_out = ['GGNN']  # ['GGNN', 'IN', 'DTNN', 'My']
architectures = ['test']  # ['ConvNet', 'RecNet', 'ResConvNet', 'resRecNet']

data_path = r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data'

train_data_f = os.path.join(data_path, 'test.txt')
test_data_f = os.path.join(data_path, 'test.txt')
test_more_data_f = os.path.join(data_path, 'v9Test.txt')

train_epochs = 100
batch_size = 64
logs_dir = '/home/alex/gsatnn'
print_rate = 10

for var_lim in var_annot_lim:
    for gt in graph_types:
        print('loading train data')
        train_data = dataset(train_data_f, gt, var_lim)
        print('loading test data')
        test_data = dataset(test_data_f, gt, var_lim)
        # test_more_data = dataset(test_more_data_f, gt, var_lim)
        all_nodes = train_data.all_nodes.copy()
        for ro in read_out:
            for arch in architectures:
                print(f'architecture: {arch}, read out: {ro}, variables similar: {var_lim == 1}, graph type: {gt}')
                net = make_net(arch, ro, gt, all_nodes, logs_dir)
                i = 0
                for ep in range(train_epochs):
                    for nodes, adjs, labels in train_data.BatchGen(batch_size):
                        net.train_step(nodes, adjs, labels, (i % print_rate) == 0)
                        i+=1
                    #net.test(test_data)
        #        net.test(test_more_data)
