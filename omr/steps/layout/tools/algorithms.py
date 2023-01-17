import PIL
from PIL import Image

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import PageScaleReference


def cga_quantize(image):
    pal_image = Image.new("P", (1, 1))
    pal_image.putpalette((207, 202, 187, 63, 63, 60, 185, 103, 91) + (207, 202, 187) * 253)
    return image.convert("RGB").quantize(palette=pal_image, dither=0)



if __name__ == "__main__":

    b = DatabaseBook('Pa_14819')
    from matplotlib import pyplot as plt
    import numpy as np
    # b = DatabaseBook('test3')
    # b = DatabaseBook('Cai_72')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[1:5]]
    for i in val_pcgts:
        path = i.page.location.file(PageScaleReference(0).file('color'), True).local_path()
        image = Image.open(path)
        image.show()

        image = cga_quantize(image)
        image.show()

        image_np = np.asarray(image)
        image_np1 = (image_np == 2)
        image_np2 = image_np * image_np1
        Image.fromarray(image_np1).show()

        plt.imshow(np.asarray(image))
        plt.show()
    pass
    #pred = DropCapitalPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    #ps = list(pred.predict([p.page.location for p in val_pcgts]))
    #print()