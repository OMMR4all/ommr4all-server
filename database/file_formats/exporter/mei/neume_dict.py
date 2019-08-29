import itertools


class NeumeDict:
    def __init__(self):
        self.enum = ['upgaped', 'downgaped', 'gaped', 'uplooped', 'downlooped', 'looped']
        self.dict_of_c_product = {}
        self.run()
        self.update()

    def run(self):
        c_product = []
        for x in range(1, 6):
            c_product += list(list(itertools.product(self.enum, repeat=x)))

        for x_ind, x in enumerate(c_product):
            c_product[x_ind] = ('neume_start', ) + x

        self.dict_of_c_product = dict.fromkeys(c_product, None)

    def update(self):
        self.dict_of_c_product[('neume_start', 'uplooped')] = 'pes'
        self.dict_of_c_product[('neume_start', 'downlooped')] = 'clivis'
        self.dict_of_c_product[('neume_start', 'upgaped', 'upgaped')] = 'scandius'

    def get_neume_type(self, key: list):
        if len(key) <= 1:
            return 'virga'
        return self.dict_of_c_product.get(tuple(key))
