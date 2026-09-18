"""Observed interaction counts plus a clearly labelled cluster-bulk simulation."""
def annotation_efficiency(store, cluster_metrics):
    n=len(store.candidates); interactions=store.interactions
    labelled=sum(i.n_affected for i in interactions)
    overrides=sum(bool(c.manual_fields) for c in store.candidates.values())
    clusters={c.cluster_id for c in store.candidates.values() if c.cluster_id!='-1'}
    # Impure members are approximated from weighted family purity among matched candidates.
    purity=(cluster_metrics or {}).get('family',{}).get('weighted_purity',0.0)
    impure=round(n*(1-purity))
    simulated=len(clusters)+impure
    return {'n_interactions':len(interactions),
            'labels_per_interaction':labelled/len(interactions) if interactions else 0.0,
            'n_instance_overrides':overrides,
            'simulated_interactions_cluster_bulk':simulated,
            'simulated_interactions_per_instance':n,
            'simulated_speedup':n/simulated if simulated else None}
