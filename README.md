
# GraphEmbedding 相关论文阅读总结

## DeepWalk

### 思路

核心思想就是使用randomwalk对graph进行采样，将获得的vertex序列当作SkipGram模型中的“句子”，构造一个语料库，丢给word2vec训练得到latent representation，最后丢给downstream的分类器去做分类。
另外，为了降低SkipGram中计算概率的复杂度，引入了一个分层softmax算法。

### 算法参数

- --number-walks, the number of random walks to start at each node; the default is 10;
- --walk-length, the length of random walk started at each node; the default is 80;paper setting: 40
- --workers, the number of parallel processes; the default is 8;
- --window-size, the window size of skip-gram model; the default is 10;
- embedding size d, paper setting: 128

## node2vec

### 思路

考虑了两种相似：homophily的相似(直观上的邻居，找距离最近的点) 和 structure的相似(结构上的相似，比如都是图中的“桥”)。改进了randomwalk的采样方法，通过参数q控制偏向BFS(q小)还是DFS(q大)的采样。其中BFS用于找structure的相似点(因为BFS会探究周围一片区域，可以发现节点的结构特征)；DFS用于找homophily的相似(因为DFS强调距离上的远近)；
另外还通过参数p控制walk时是否访问已经访问过的点的概率,p小的时候倾向于访问已经访问过的点，即探索更local；p大的时候倾向于探索未访问过的节点。
当p=q=1时等价于deepwalk(因此，openNE事实上只实现了node2vec算法)

### 算法参数

- --number-walks, the number of random walks to start at each node; the default is 10;
- --walk-length, the length of random walk started at each node; the default is 80;paper setting: 40
- --workers, the number of parallel processes; the default is 8;
- --window-size, the window size of skip-gram model; the default is 10;
- embedding size d, paper setting: 128
- return parameter p (0.25 0.5 1 2 4) 
- in-out parameter q (0.25 0.5 1 2 4)


## TADW

### 思路

- 证明了DeepWalk其实是等价于矩阵分解。证明分两个部分，第一部分是直接给出一个结论，deepwalk借鉴的使用softmax的skip-gram是分解word-context矩阵M；第二部分是讨论了$$M_{ij}$$的意义，即顶点V_i随机游走固定步数能到达顶点V_j的期望概率的对数，最后利用pagerank的转移矩阵给出了如何计算这个概率
- 基于上述矩阵分解的思想，重新定义了优化函数(比较复杂)，引入了harmonic factor \\(\lambda\\)，提出了Text-Associated DeepWalk(TADW)算法，在网络结构这一feature的基础上又添加了节点的文本特征；即给定一个网络G=(V，E)和相应的文本特征矩阵T(对TFIDF矩阵进行SVD降维得到)，TADW从G和T中去学习每个节点的表示方式。
- 一个拓展方向:在矩阵分解的框架下，添加其他的特征

### 算法参数

表示向量维度2k和 \lambda (hyperparameter in TADW that controls the weight of regularization term)


## GraRep

### 思路
- 引入了一个超参数k来表示当前节点与距离当前节点k步的节点的关系
- 作者通过描述网络的邻接矩阵和对应的顶点的出度矩阵，来定义了一个转移矩阵(和TADW的转移矩阵很类似)，也同样利用矩阵的乘幂来表示走了k步。然后借用NCE(noise contrastive estimation)来定义要优化的目标函数。在优化目标函数时，使用了矩阵分解(SVD)来得到第k步的representation vector，最后将k个representation vector拼在一起作为网络整体的representation vector
- 最后证明了skip gram with negative sampling是GraRep的特例。这和TADW证明deepwalk的random walk是一种矩阵分解不谋而合。
- 矩阵的观点

### 算法参数

- 超参数k来表示当前节点与距离当前节点k步的节点的关系
(k=6 for blogcatalog) 
- openne的说明:--kstep, use k-step transition probability matrix（make sure representation-size%k-step == 0).


## LINE

### 思路
引入了一阶和二阶的相似度概念：
    - 一阶：直接相连
    - 二阶：邻居类似
然后用KL散度针对一阶和二阶的相似度设计了目标函数；二阶相似度考虑了条件概率密度和边的权重；最后将一阶feature和二阶feature拼在一起作为LINE提出来的feature
为了大量级运算的优化，用负采样的方法，来进行权重更新；还有为了避免边的权重的方差过大导致的难以设置学习率的问题，基于边的权值大小来设置采样的概率，对边也进行了采样。


### 算法参数
--negative-ratio, 负采样的样本个数　the default is 5;
--order, 1 for the 1st-order, 2 for the 2nd-order and 3 for 1st + 2nd; the default is 3;
--no-auto-save, no early save when training LINE; this is an action; when training LINE, we will calculate F1 scores every epoch. If current F1 is the best F1, the embeddings will be saved.


## SDNE

### 思路
为了有效捕捉高度非线性网络结构并保持全局以及局部的结构，作者在论文中提出了Structural Deep Network Embedding (SDNE) 方法。论文首次提出了一个半监督的深层模型，它具有多层非线性函数，从而能够捕获到高度非线性的网络结构。然后使用一阶和二阶邻近关系来保持网络结构。二阶邻近关系被无监督学习用来捕获全局的网络结构，一阶邻近关系使用监督学习来保留网络的局部结构。通过在半监督深度模型中联合优化它们，该方法可以保留局部和全局网络结构
- 用了deep autoencoder 通过reconstruction来保持二阶邻近关系，同时有个加权B, Xi-Xj
- 用监督方法来保持一阶邻近关系，Yi-Yj
- 最后再加上防止过拟合的正则项联合优化
``
 For each vertex, we are able to obtain its neighborhood. Accordingly, we design the unsupervised component to preserve the second-order proximity, by reconstructing the neighborhood structure of each vertex. Meanwhile, for a small portion of pairs of nodes, we can obtain their pairwise similarities, i.e. the ﬁrst-order proximities. Therefore, we design the supervised component to exploit the ﬁrst-order proximity as the supervised information to reﬁne the representations in the latent space. By jointly optimizing them in the proposed semi-supervised deep model, SDNE can preserve the highly-nonlinear local-global networkstructurewellandisrobusttosparsenetworks. 
 ``

When α = 0, the performance is totally determined by the second-order proximity. And the larger the α, the more the model concentratesontheﬁrst-orderproximity


### 算法参数

--encoder-list, a list of numbers of the neuron at each encoder layer, the last number is the dimension of the output node representation, the default is [1000, 128]

--alpha, alpha is a hyperparameter in SDNE that controls the first order proximity loss, the default is 1e-6

--beta, beta is used for construct matrix B, the default is 5

--nu1, parameter controls l1-loss of weights in autoencoder, the default is 1e-5

--nu2, parameter controls l2-loss of weights in autoencoder, the default is 1e-4

--bs, batch size, the default is 200



--lr, learning rate, the default is 0.01


## HOPE

### 思路
这个算法主要使用 high-order proximity preserved embedding (HOPE) method来解决有向图的embedding。为了避免矩阵分解的运算复杂度，他们使用了广义的SVD分解。解决边的方向问题的核心想法是，每个点学两个embedding，一个source，一个target，如果有从点v到点u的边，那么v的source embedding 和u的target embedding值就很接近，如果没有从u到v的边，那么u的source embedding 和v的target embedding值就差别很大


### 算法参数

就是矩阵分解，唯一的参数就是是embedding的维度，值得注意的是在OpenNe测试HOPE时输入的维度是source embedding 和 target embedding 的维度之和,即128=64source+64target


## node2hash
*Feature Hashing for Network Representation Learning*

### 思路
这个算法的创新点有两个，一个是使用了新的proximity matrix（size也是|V|*|V|）来表示节点的位置分布和co-occurence概率，这个矩阵的构造仍基于random walk，作者自己定义了expected distance的概念，然后将所有节点编号为1到|V|，将window中两个共同出现的节点的expected distance之和作为矩阵中特定位置（由两个节点的标号决定）的值；第二个创新点就是提出使用feature hashing来对构造出的proximity矩阵进行降维；从实验结果上来看，他们的算法在节点关系预测(link prediction)上表现不错，在多标签分类任务上，表现一般，不过他们抽出来的proximity matrix表现还挺好；这篇论文比较值得借鉴和移植的就是他们的proximity matrix，这个是个挺框架性的东西，可以在一些基于矩阵分解的NE算法中进行尝试

### 算法参数

window size; hash函数数目; latent dimension; walk-length; number of walks

## Self-Paced Network Embedding
### 思路
使用self-pace的思想改良了负采样的过程. 

第一种方法是用softmax定义选择负采样的节点的概率分布,而不是像LINE那样基于节点度数.然后使用self-pace的思想,先选择简单的负样本开始训练,再逐渐选择困难的负样本进行训练(这里的"简单"与"困难"指的是如果一个负样本多次迭代后在embedding space里离中心节点比较近,则可以认为将这个节点与中心节点分开比较困难,所以我们直觉地可以认为它可能包含更多的信息).self-pace的实现其实就是加了一个阈值,本身困难的样本被选择的概率比较大,通过设置这个阈值,每次选小于阈值的样本,这样可以选一些简单的样本,而阈值逐渐增大,也就更倾向于选择困难的样本.

论文第二种方法是套了个GAN,用一个generator生成的概率分布来采样负样本,这个generator会训练一个新的embedding vector(?),然后同样套上一个self-pace的阈值函数来选择样本.