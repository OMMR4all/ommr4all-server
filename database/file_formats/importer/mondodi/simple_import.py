class MonodiDocument:
    def __init__(self, sentence):
        self.sentence = sentence

    def get_word_list(self, text_normalizer):
        words = []
        word = ""
        for x in self.sentence:
            if len(x) > 0:
                if x[-1] == "-":
                    word = word + x[:-1]
                elif x == "\n" or x == "\n\n":
                    pass
                else:
                    word = word + x
                    words.append(word)
                    word = ''
        return words

    def get_text(self, text_normalizer=None):
        text = ""
        for x in self.sentence:
            if len(x) > 0:
                text += x
                if x[-1] == "-":
                    pass
                else:
                    text += " "
        return text_normalizer.apply(text) if text_normalizer else text


def simple_monodi_data_importer1(json):
    sentence = []
    for x in json["children"]:
        for y in x["children"]:
            if y["kind"] == "ZeileContainer":
                for z in y["children"]:
                    if z["kind"] == "Syllable":
                        sentence.append(z["text"])
                    elif z["kind"] == "LineChange":
                        pass
                        # sentence.append("\n")
                    elif z["kind"] == "FolioChange":
                        pass
                        # sentence.append("\n")
                    else:
                        pass
                        # print(z["kind"])
    return MonodiDocument(sentence)


def getRowContainer(dict, list):
    if "children" in dict:
        for x in dict["children"]:
            if "kind" in x and x["kind"] != "ZeileContainer":
                getRowContainer(x, list=list)
            else:
                list.append(x)
    else:
        pass
        # print(dict)


def simple_monodi_data_importer(json):
    sentence = []
    row_container = []
    getRowContainer(json, row_container)
    for x in row_container:
        for z in x["children"]:
            if z["kind"] == "Syllable":
                sentence.append(z["text"])
            elif z["kind"] == "LineChange":
                pass
                sentence.append("\n")
            elif z["kind"] == "FolioChange":
                pass
                sentence.append("\n\n")
            else:
                pass
    return MonodiDocument(sentence)
