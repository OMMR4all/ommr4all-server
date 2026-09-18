"""Registry of neume grouping methods."""
from typing import Dict, Type

from omr.discovery.grouping.base import NeumeGroupingMethod
from omr.discovery.grouping.rule_graph import RuleGraphNeumeGrouping

NEUME_GROUPING_METHODS: Dict[str, Type[NeumeGroupingMethod]] = {
    RuleGraphNeumeGrouping.name: RuleGraphNeumeGrouping,
}


def build_grouping_method(name: str) -> NeumeGroupingMethod:
    try: return NEUME_GROUPING_METHODS[name]()
    except KeyError:
        raise ValueError('unknown grouping method {!r} (known: {})'.format(
            name, ', '.join(sorted(NEUME_GROUPING_METHODS))))


__all__ = ['NEUME_GROUPING_METHODS', 'build_grouping_method', 'NeumeGroupingMethod',
           'RuleGraphNeumeGrouping']
