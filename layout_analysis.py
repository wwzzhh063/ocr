from inference import set_xml_data
import config

class Layout_Cell(object):
    def __init__(self,bbox,label,type):
        self.bbox = bbox
        self.label = label
        self.type = type
        self.row = -1
        self.Column = -1


class Layoout_Row(object):
    def __init__(self):
        pass



class Layput_Column(object):
    def __init__(self):
        pass


class Layput_Analysis(object):
    def __init__(self):
        self.print = []
        self.hand = []









if __name__ == '__main__':


    xml_path = config.DATA_XML
    all_img = set_xml_data(xml_path)