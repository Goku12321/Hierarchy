import anytree
from anytree import Node, RenderTree, LevelGroupOrderIter, PreOrderIter
import torch

class Tree():
    def __init__(self, data):
        if type(data) == dict:
            data = make_tree_from_nested_dict(data)
            self.root, self.data = data[0], data[1:]
        #             self.siblings = make_siblings()
        self.siblings = {}
        self.tree_path = {}
        for i,node in enumerate(self.data):
            node.index = i

        self.root.index=-1
        for i,node in enumerate(self.data):
            self.siblings[i] = [i] + [n.index for n in node.siblings]
            self.tree_path[i] = [n.index for n in node.path[1:]]

        self.siblings_groups = []
        for node in PreOrderIter(self.root):
            if len(node.children) > 0:
                self.siblings_groups.append([n.index for n in node.children])

        self.level_indices = []
        for i, level in enumerate(LevelGroupOrderIter(self.root)):
            if i == 0:
                continue
            self.level_indices.append([node.index for node in level])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"Tree | {len(self.data)} Nodes | {len(self.level_indices)} levels"

    def __str__(self):
        return self.get_tree()


    def make_hier_labels(self, labels):
        hier_labels = -torch.ones([len(labels), len(self.level_indices)])
        for i, path in enumerate([[n.index for n in self.data[L].path][1:] for L in labels]):
            hier_labels[i, :len(path)] = torch.tensor(path)
        return hier_labels

    def one_hot_vector(self, target):
        hot_vector = torch.zeros(len(target), self.__len__())
        for i, t in enumerate(target):
            hot_vector[i, self.tree_path[t.item()]] = 1
        return hot_vector

    def probs_by_outputs(self, probs_cond_batch):
        probs_batch = torch.zeros(probs_cond_batch.size())
        for batch_index, probs in enumerate(probs_cond_batch):
            for i, (pre, fill, node) in enumerate(RenderTree(self.root)):
                if i == 0:
                    node.p = 1
                else:
                    parent = node.ancestors[-1]
                    node.p = probs[node.index].item() * parent.p
                    probs_batch[batch_index, node.index] = node.p
        return probs_batch


    def get_node_by_name(self, name):
        '''
        input: name of a node in the tree
        output: Node object of the node name
        '''
        for n in self.data:
            if n.name == name:
                return n
        raise ValueError(f"'{name}' is not in the Tree stracture")

    def get_siblings_names_by_name(self, name):
        '''
        input: name of a node in the tree
        output: list of strings of the node silbings and itself
        '''
        for node in self.data:
            if node.name == name:
                return [name] + [n.name for n in node.siblings]
        raise ValueError(f"'{name}' is not in the Tree stracture")

    def get_siblings_ids_by_name(self, name):
        '''
        input: name of a node in the tree
        output: list indices of the node silbings and itself
        '''
        for node in self.data:
            if node.name == name:
                return [name] + [n.name for n in node.siblings]
        raise ValueError(f"'{name}' is not in the Tree stracture")

    def get_tree(self, show_ind=True, show_p=False):
        """
        print the tree
        """
        tree_str = ""
        for i, (pre, fill, node) in enumerate(RenderTree(self.root)):
            if i > 0:
                if show_ind:
                    tree_str = tree_str + ("%s%s %s\n" % (pre, node.name, i - 1))
                elif show_p:
                    tree_str = tree_str + ("%s%s %s: %.2f\n" % (pre, node.name, node.index, node.prob[0].item())) # index 0 for example of the first in the batch
                else:
                    tree_str = tree_str + ("%s%s\n" % (pre, node.name))
            else:
                tree_str = tree_str + ("%s%s\n" % (pre, node.name))
        return tree_str

    def show_tree(self, show_ind=True, show_p=False):
        print(self.get_tree(show_ind=show_ind, show_p=show_p))

    def show_indices(self):
        '''
        show the indices and names of all nodes
        '''
        for i, n in enumerate(self.data):
            print(f"{i}: {n.name}")

    def show_tree_path(self):
        '''
        print for each node the names of the nodes of it's path
        '''
        for n in self.data:
            print(n.name, ":", [i.name for i in n.path])

    def show_siblings(self):
        '''
        print for each node the names of its siblings
        '''
        for n in self.data:
            print(n.name, ":", [i.name for i in n.siblings])


def make_tree_from_nested_dict(d):
    global global_ind
    nodes = []
    global_ind = -1

    def _get_dict_item(d, parent='', parent_id=0):
        global global_ind
        l = []; n = [];
        if type(d) is dict:
    #         print(d.keys())
            keys, items = zip(*sorted(d.items(), key = lambda x:x[0]))
    #         print(parent,":", keys)
            l += list(keys)
            for i,(key, item) in enumerate(zip(keys, items)):
                global_ind += 1
#                 print(global_ind, key, parent, parent_id)
                if global_ind > 0:
                    nodes.append(Node(key, nodes[parent_id]))
                else:
                    nodes.append(Node(key))
                _l, _n = _get_dict_item(item, key, global_ind)
                l += [key + '/' +i for i in _l]
                n += [_n]
        else:
            return d, 0
        return l, n
    _get_dict_item(d)
    return nodes


class CIFFAR100_HIER_CLASSES():
    def __init__(self):
        super_classes = ["aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
                         "household_electrical_devices"
            , "household_furniture", "insects", "large_carnivores", "large_man-made_outdoor_things",
                         "large_natural_outdoor_scenes"
            , "large_omnivores_and_herbivores", "medium_mammals", "non-insect_invertebrates", "people", "reptiles",
                         "small_mammals", "trees", "vehicles_1", "vehicles_2"]

        classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                   ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                   ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                   ['bottle', 'bowl', 'can', 'cup', 'plate'],
                   ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                   ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                   ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                   ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                   ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                   ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                   ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                   ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                   ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                   ['crab', 'lobster', 'snail', 'spider', 'worm'],
                   ['baby', 'boy', 'girl', 'man', 'woman'],
                   ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                   ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                   ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                   ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                   ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

        cifar100_dict = {}
        for sc, sc_cl in zip(super_classes, classes):
            cifar100_dict[sc] = {cl: "" for cl in sc_cl}
        cifar100_dict = {'object': cifar100_dict}
        self.cifar100_dict = cifar100_dict



