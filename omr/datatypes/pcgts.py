from omr.datatypes import Meta, Page


class PcGts:
    def __init__(self, meta=Meta(), page=Page()):
        self.meta = meta
        self.page = page

    @staticmethod
    def from_file(filename):
        if filename.endswith(".json"):
            import json
            with open(filename, 'r') as f:
                return PcGts.from_json(json.load(f))

        else:
            raise Exception("Invalid file extension of file '{}'".format(filename))

    def to_file(self, filename):
        if filename.endswith(".json"):
            import json
            with open(filename, 'w') as f:
                json.dump(self.to_json(), f, indent=2)
        else:
            raise Exception("Invalid file extension of file '{}'".format(filename))

    @staticmethod
    def from_json(json: dict):
        return PcGts(
            Meta.from_json(json.get('meta', {})),
            Page.from_json(json.get('page', {})),
        )

    def to_json(self):
        return {
            'meta': self.meta.to_json(),
            'page': self.page.to_json(),
        }


if __name__ == '__main__':
    from omr.datatypes import *
    pcgts = PcGts(Meta(), Page(
        [
            TextRegion(
                TextRegionType.LYRICS
            )
        ]
    ))

    print(pcgts.to_json())
    print(PcGts.from_json(pcgts.to_json()).to_json() == pcgts.to_json())




