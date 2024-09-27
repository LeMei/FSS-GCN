FSS-GCN: A graph convolutional networks with fusion of semantic and structure for emotion cause analysis
===========
Abstract
---

Most existing methods capture semantic information by using attention mechanism or joint learn- ing, ignoring inter-clause dependency. However, inter-clause dependency contains richer structural information that is useful to alleviate information loss so as to understand text deeply. To tackle this problem, we construct a graph suited to encode dependency information at clause level. Based on this graph, we propose a graph convolutional network over the inter-clause dependency to fuse the semantics and structural information, which automatically learns how to selectively attend the relevant clauses useful for emotion cause analysis. Intuitively, our model can be understood as a method that narrows focus from global structure to local structure by continuously injecting structural constricts into networks. Our proposed method is evaluated on two public datasets in different languages (Chinese and English). Experimental results demonstrate that our model achieves superior performance compared to the existing methods, in which the graph convolution structure is found to be effective for emotion cause analysis. We further conduct experiments to confirm the ability of our model to capture long-distance information in terms of semantic and structural information.


Full codes (including raw and preprocessing data): https://drive.google.com/drive/folders/1DDOHnP28Frq3I_cjZpAWNm3N1zxpF4Su?usp=sharing

If you use the codes, please cite our work. Thanks in advance!
