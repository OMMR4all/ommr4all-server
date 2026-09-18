"""Purity, information metrics and nearest-neighbour retrieval quality."""
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _purity(labels, truth):
    groups=defaultdict(list)
    for label,value in zip(labels,truth): groups[label].append(value)
    return sum(max(Counter(v).values()) for v in groups.values())/len(labels) if labels else 0.0


def _retrieval(embeddings, truth, ks):
    if len(embeddings)<2: return {'precision@{}'.format(k):0.0 for k in ks}
    x=np.asarray(embeddings,float); x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    sim=x@x.T; np.fill_diagonal(sim,-np.inf)
    order=np.argsort(-sim,axis=1)
    return {'precision@{}'.format(k):float(np.mean([
        np.mean([truth[j]==truth[i] for j in order[i,:min(k,len(x)-1)]])
        for i in range(len(x))])) for k in ks}


def _one(labels, truth, embeddings, ks):
    keep=[i for i,label in enumerate(labels) if label!='-1']
    if not keep: return {'weighted_purity':0.0,'nmi':0.0,'ari':0.0,**_retrieval(np.zeros((0,1)),[],ks)}
    l=[labels[i] for i in keep]; t=[truth[i] for i in keep]; e=np.asarray(embeddings)[keep]
    return {'weighted_purity':_purity(l,t),
            'nmi':float(normalized_mutual_info_score(t,l)),
            'ari':float(adjusted_rand_score(t,l)),**_retrieval(e,t,ks)}


def cluster_metrics(labels, gt_families, gt_subtypes, embeddings, ks):
    outliers=[i for i,label in enumerate(labels) if label=='-1']
    return {'family':_one(labels,gt_families,embeddings,ks),
            'subtype':_one(labels,gt_subtypes,embeddings,ks),
            'outlier_rate':len(outliers)/len(labels) if labels else 0.0,
            'outlier_family_histogram':dict(Counter(gt_families[i] for i in outliers)),
            'n_clusters':len({x for x in labels if x!='-1'}),'n_outliers':len(outliers)}
