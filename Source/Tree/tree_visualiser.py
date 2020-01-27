''' Generates visual representations of game trees.
@classmod tree_visualiser'''
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Game.card_to_string_conversion import card_to_string
from graphviz import render

# TODO: README
# dot tree_2.dot -Tpng -O

class Graph:
    def __init__(self):
        super().__init__()
        self.label = None
        self.name = None
        self.shape = None

class Node:
    def __init__(self):
        super().__init__()
        self.id_from = None
        self.id_to = None
        self.id = None
        self.strategy = None

class TreeVisualiser:
    def __init__(self):
        super().__init__()
        self.node_to_graphviz_counter = 0
        self.edge_to_graphviz_counter = 0

    def add_tensor(self, tensor, name=None, _format=None, labels=None):
        ''' Generates a string representation of a tensor.
        @param tensor a tensor
        @param[opt] name a name for the tensor
        @param[opt] format a format string to use with @{string.format} for each
        element of the tensor
        @param[opt] labels a list of labels for the elements of the tensor
        @return a string representation of the tensor
        @local'''
        out = ''
        if name:
            out = '| ' + name + ': '
        
        if not _format:
            _format = '{:.3f}'

        for i in range(tensor.size(0)):
            if labels:
                out = out + labels[i] + ":"
            out = out + _format.format(tensor[i]) + ", "
        
        return out

    def add_range_info(self, node):
        ''' Generates a string representation of any range or value fields that are set
        for the given tree node.
        @param node the node
        @return a string containing concatenated representations of any tensors
        stored in the `ranges_absolute`, `cf_values`, or `cf_values_br` fields of
        the node.
        @local'''   
        out = ""
        
        if(node.ranges_absolute != None):
            out = out + self.add_tensor(node.ranges_absolute[0], 'abs_range1')
            out = out + self.add_tensor(node.ranges_absolute[1], 'abs_range2')

        if(node.cf_values != None):
            # cf values computed by real tree dfs
            out = out + self.add_tensor(node.cf_values[0], 'cf_values1')
            out = out + self.add_tensor(node.cf_values[1], 'cf_values2')
        
        if(node.cf_values_br != None):
            # cf values that br has in real tree
            out = out + self.add_tensor(node.cf_values_br[0], 'cf_values_br1')
            out = out + self.add_tensor(node.cf_values_br[1], 'cf_values_br2')
        
        return out

    def node_to_graphviz(self, node):  
        ''' Generates data for a graphical representation of a node in a public tree.
        @param node the node to generate data for
        @return a table containing `name`, `label`, and `shape` fields for graphviz
        @local''' 
        out = Graph()
        
        # 1.0 label
        out.label = '"<f0>' + str(node.current_player)
        
        if node.terminal:
            if node.type == constants.node_types.terminal_fold:
                out.label = out.label + '| TERMINAL FOLD'
            elif node.type == constants.node_types.terminal_call:
                out.label = out.label + '| TERMINAL CALL'
            else:
                assert False, 'unknown terminal node type'
        else:
            out.label = out.label + '| bet1: ' + str(node.bets[constants.players.P1].item()) + '| bet2: ' + str(node.bets[constants.players.P2].item())
            
            if node.street:
                out.label = out.label + '| street: ' + str(node.street)
                out.label = out.label + '| board: ' + card_to_string.cards_to_string(node.board)
                out.label = out.label + '| depth: ' + str(node.depth)
        
        if node.margin:
            out.label = out.label +  '| margin: ' + node.margin

        out.label = out.label + self.add_range_info(node)  
        
        if(node.cfv_infset != None):
            out.label = out.label +  '| cfv1: ' + str(node.cfv_infset[0].item())
            out.label = out.label +  '| cfv2: ' + str(node.cfv_infset[1].item())
            out.label = out.label +  '| cfv_br1: ' + str(node.cfv_br_infset[0].item())
            out.label = out.label +  '| cfv_br2: ' + str(node.cfv_br_infset[1].item())
            out.label = out.label +  '| epsilon1: ' + str(node.epsilon[0].item())
            out.label = out.label +  '| epsilon2: ' + str(node.epsilon[1].item())
        
        if node.lookahead_coordinates:
            out.label = out.label +  '| COORDINATES '
            out.label = out.label +  '| action_id: ' + str(node.lookahead_coordinates[0].item())
            out.label = out.label +  '| parent_action_id: ' + str(node.lookahead_coordinates[1].item())
            out.label = out.label +  '| gp_id: ' + str(node.lookahead_coordinates[2].item())
        
        out.label = out.label + '"'
        
        # 2.0 name
        out.name = '"node' + str(self.node_to_graphviz_counter) + '"'
        
        # 3.0 shape
        out.shape = '"record"' 
            
        self.node_to_graphviz_counter = self.node_to_graphviz_counter + 1
        return out

    def nodes_to_graphviz_edge(self, _from, to, node, child_node):
        ''' Generates data for graphical representation of a public tree action as an
        edge in a tree.
        @param from the graphical node the edge comes from
        @param to the graphical node the edge goes to
        @param node the public tree node before at which the action is taken
        @param child_node the public tree node that results from taking the action
        @return a table containing fields `id_from`, `id_to`, `id` for graphviz and
        a `strategy` field to use as a label for the edge
        @local'''
        out = Node()
        
        out.id_from = _from.name
        out.id_to = to.name
        out.id = self.edge_to_graphviz_counter
        
        # get the child id of the child node
        child_id = -1
        for i in range(len(node.children)):
            if node.children[i] == child_node:
                child_id = i
        
        assert(child_id != -1)
        out.strategy = self.add_tensor(node.strategy[child_id], None, '{:.2f}', card_to_string.card_to_string_table)
        
        self.edge_to_graphviz_counter = self.edge_to_graphviz_counter + 1
        return out

    def graphviz_dfs(self, node, nodes, edges):
        ''' Recursively generates graphviz data from a public tree.
        @param node the current node in the public tree
        @param nodes a table of graphical nodes generated so far
        @param edges a table of graphical edges generated so far
        @local'''
        gv_node = self.node_to_graphviz(node)
        nodes.append(gv_node)
        
        for i in range(len(node.children)):
            child_node = node.children[i]
            gv_node_child = self.graphviz_dfs(child_node, nodes, edges)
            gv_edge = self.nodes_to_graphviz_edge(gv_node, gv_node_child, node, child_node)
            edges.append(gv_edge)

        return gv_node

    def graphviz(self, root, filename='tree_2.dot'):
        ''' Generates `.dot` and `.svg` image files which graphically represent 
        a game's public tree.

        Each node in the image lists the acting player, the number of chips
        committed by each player, the current betting round, public cards,
        and the depth of the subtree after the node, as well as any probabilities
        or values stored in the `ranges_absolute`, `cf_values`, or `cf_values_br`
        fields of the node.

        Each edge in the image lists the probability of the action being taken
        with each private card.
        # 
        @param root the root of the game's public tree
        @param filename a name used for the output files'''
        out = 'digraph g {  graph [ rankdir = "LR"];node [fontsize = "16" shape = "ellipse"]; edge [];'
            
        nodes = []
        edges = []
        self.graphviz_dfs(root, nodes, edges)
            
        for i in range(len(nodes)):
            node = nodes[i]
            node_text = node.name + '[' + 'label=' + node.label + ' shape = ' + node.shape + '];'
            
            out = out + node_text
            
        for i in range(len(edges)):
            edge = edges[i]
            edge_text = edge.id_from + ':f0 -> ' + edge.id_to + ':f0 [ id = ' + str(edge.id) + ' label = "' + edge.strategy + '"];'
            
            out = out + edge_text
            
        out = out + '}'
            
        # write into dot file
        filename = arguments.project_root + "/Data/Dot/" + filename + '.dot'
        with open(filename, 'w') as f:
            f.write(out)

        render('dot', 'png', filename)