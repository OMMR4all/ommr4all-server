from omr.imageoperations import ImageOperationData


class RegionLineMaskData:
    def __init__(self, op: ImageOperationData):
        self.operation = op
        self.line_image = op.images[1].image
        self.region = op.images[0].image
        self.mask = op.images[2].image

