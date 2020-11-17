"""
The GripNet is an efficient framework to learn node representations on heterogeneous graphs for the
downstream link prediction, node classification, and visualization. The code is based on
the `GripNet
<https://github.com/NYXFLOWER/GripNet>`_ source repo.
"""

import torch.nn.functional as F
import torch
from torch.nn import Module
from kale.embed.gcn import GCNEncoderLayer, RGCNEncoderLayer


# Copy-paste with slight modification from https://github.com/NYXFLOWER/GripNet
class GripNetSuperVertex(Module):
    r"""
    The supervertex module in GripNet. Each supervertex is a subgraph containing nodes with the
    same category or at least keep semantically-coherent. The supervertex can be regarded as homogeneous graph and
    information is propagated between them.

    Args:
        nhid_list (list): Dimensions list of hidden layers e.g. [hidden_1, hidden_2, ... hidden_n]
        requires_grad (bool, optional): Requires gradient for initial embedding (default :obj:`True`)
        start_graph (bool, optional): If set to :obj:`True`, this supervertex is the
            start point of the whole information propagation. (default :obj:`False`)
        in_dim (int, optional): the size of input sample for start graph. (default :obj:`None`)
        multi_relational: If set to :obj: 'True', the supervertex is a multi relation graph. (default :obj:`False`)
        n_rela (int, optional): Number of edge relations if supervertex is a multi relation graph. (default :obj:`None`)
        n_base (int, optional): Number of bases if supervertex is a multi relation graph. (default :obj:`None`)
    """

    def __init__(self, nhid_list, requires_grad=True, start_graph=False,
                 in_dim=None, multi_relational=False, n_rela=None, n_base=32):
        super(GripNetSuperVertex, self).__init__()
        self.multi_relational = multi_relational
        self.start_graph = start_graph
        self.out_dim = nhid_list[-1]
        self.n_cov = len(nhid_list) - 1

        if start_graph:
            self.embedding = torch.nn.Parameter(torch.Tensor(in_dim, nhid_list[0]))
            self.embedding.requires_grad = requires_grad
            self.reset_parameters()

        if multi_relational:
            assert n_rela is not None
            after_relu = [False if i == 0 else True for i in
                          range(len(nhid_list) - 1)]
            self.conv_list = torch.nn.ModuleList([
                RGCNEncoderLayer(nhid_list[i], nhid_list[i + 1], n_rela, n_base, after_relu[i])
                for i in range(len(nhid_list) - 1)])
        else:
            self.conv_list = torch.nn.ModuleList([
                GCNEncoderLayer(nhid_list[i], nhid_list[i + 1], cached=True)
                for i in range(len(nhid_list) - 1)])

    def reset_parameters(self):
        self.embedding.data.normal_()

    def forward(self, x, homo_edge_index, edge_weight=None, edge_type=None,
                range_list=None, if_catout=False):
        """
        Args:
            x (torch.Tensor): the input node feature embedding.
            homo_edge_index (torch.Tensor): edge index in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor): one-dimensional relation weight for each edge.
            edge_type (torch.Tensor): one-dimensional relation type for each edge in.
            range_list (list): the index range list of each edge type with shape [num_types, 2].
            if_catout (bool): whether to concatenate each layer's output.
        """
        if self.start_graph:
            x = self.embedding

        if if_catout:
            tmp = []
            tmp.append(x)

        if self.multi_relational:
            assert edge_type is not None
            assert range_list is not None

        for net in self.conv_list[:-1]:
            x = net(x, homo_edge_index, edge_type, range_list) \
                if self.multi_relational \
                else net(x, homo_edge_index, edge_weight)
            x = F.relu(x, inplace=True)
            if if_catout:
                tmp.append(x)

        x = self.conv_list[-1](x, homo_edge_index, edge_type, range_list) \
            if self.multi_relational \
            else self.conv_list[-1](x, homo_edge_index, edge_weight)

        x = F.relu(x, inplace=True)
        if if_catout:
            tmp.append(x)
            x = torch.cat(tmp, dim=1)
        return x


class GripNetSuperEdges(Module):
    r"""
    The superedges module in GripNet. Each superedges is a bipartite subgraph containing nodes from two categories
    forming two nodes set, connected by edges between them. The superedge can be regards as a heterogeneous graph
    connecting different supervertexs. It achieves efficient information flow propagation from all parents supervetices
    to target supervertex.

    Args:
        source_dim (int): Embedding dimensions of each source node.
        target_dim (int): Embedding dimensions of each target node aggregated from source nodes.
        n_target (int): Numbers of target nodes.
        target_feat_dim (int, optional): Initial dimensions of each target node for internal layer. (default: 32)
        requires_grad (bool, optional): Require gradient for the part of initial target node embedding.
            (default: :obj:`True`)
    """

    def __init__(self, source_dim, target_dim, n_target, target_feat_dim=32,
                 requires_grad=True):
        super(GripNetSuperEdges, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.target_feat_dim = target_feat_dim
        self.n_target = n_target
        self.target_feat = torch.nn.Parameter(
            torch.Tensor(n_target, target_feat_dim))

        self.target_feat.requires_grad = requires_grad

        self.conv = GCNEncoderLayer(source_dim, target_dim, cached=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.target_feat.data.normal_()

    def forward(self, x, inter_edge_index, edge_weight=None, if_relu=True, mod='cat'):
        """
        Args:
            x (torch.Tensor): the input node feature embedding.
            inter_edge_index (torch.Tensor): edge index in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor): one-dimensional relation weight for each edge.
            if_relu (bool): use relu function or not.
            mod (string): the aggregation schema to use (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        """
        n_source = x.shape[0]
        tmp = inter_edge_index + 0
        tmp[1, :] += n_source

        x = torch.cat(
            [x, torch.zeros((self.n_target, x.shape[1])).to(x.device)], dim=0)
        x = self.conv(x, tmp, edge_weight)[n_source:, :]
        if if_relu:
            x = F.relu(x)
        if mod == 'cat':
            x = torch.cat([x, torch.abs(self.target_feat)], dim=1)
        else:
            assert x.shape[1] == self.target_feat.shape[1]
            x = x + torch.abs(self.target_feat)
        return x


class TypicalGripNetEncoder(Module):
    r"""
    A typical GripNet architecture with one external aggregation feature layer (GCNs) and one internal layer (RGCNs).
    The information propagates from one source nodes set to one target nodes set. You can also
    define self topological ordering of the supervertices the specific graph belongs to. For more details about GripNet,
    please see `"GripNet"
    <https://github.com/NYXFLOWER/GripNet>`_ repo.

    Args:
        source_nhids (list): Dimensions list of source nodes' hidden layers e.g. [hidden_1, hidden_2, ... hidden_n]
        st_inter_out (list): Dimensions list of superedge between source and target node sets with length 2.
        target_nhids (list): Dimensions list of target nodes' hidden layers e.g. [hidden_1, hidden_2, ... hidden_n]
        n_target_node (int): Numbers of target nodes.
        n_source_node (int): Numbers of source nodes.
        n_target_edge_type (int): Number of edge relations of target supervertex.
    """

    def __init__(self, source_nhids, st_inter_out, target_nhids, n_target_node, n_source_node, n_target_edge_type):
        super(TypicalGripNetEncoder, self).__init__()
        self.n_target_node = n_target_node
        self.n_source_node = n_source_node
        self.source_graph = GripNetSuperVertex(source_nhids, start_graph=True, in_dim=self.n_source_node)
        self.s2t_graph = GripNetSuperEdges(sum(source_nhids), st_inter_out[0],
                                           self.n_target_node, target_feat_dim=st_inter_out[-1])
        self.target_graph = GripNetSuperVertex(target_nhids, multi_relational=True, n_rela=n_target_edge_type)

    def forward(self, source_feat, source_edge_index, source_edge_weight,
                st_edge_index, target_edge_idx, target_edge_et, target_edge_range):
        """
        Args:
            source_feat (torch.Tensor): Input source node feature embedding.
            source_edge_index (torch.Tensor): Source edge index in COO format with shape [2, num_edges].
            source_edge_weight (torch.Tensor): The one-dimensional relation weight
                for each edge in source graph.
            st_edge_index: Source-target edge index in COO format with shape [2, num_edges].
            target_edge_idx: Target edge index in COO format with shape [2, num_edges].
            target_edge_et: The one-dimensional relation type/index for each target edge in
                :obj:`edge_index`.
            target_edge_range: The index range list of each target edge type with shape [num_types, 2].
        """
        z = self.source_graph(source_feat, source_edge_index, edge_weight=source_edge_weight, if_catout=True)
        z = self.s2t_graph(z, st_edge_index)
        z = self.target_graph(z, target_edge_idx, edge_type=target_edge_et,
                              range_list=target_edge_range, if_catout=True)
        return z