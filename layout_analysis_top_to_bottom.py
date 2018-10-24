from inference import set_xml_data,draw_bbox
from config import Config as config
import cv2
from tqdm import tqdm

class Layout_Cell(object):
    def __init__(self,bbox,label,type):
        self.bbox = bbox
        self.label = label
        self.type = type
        self.row = -1
        self.Column = -1
        self.centre = ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)

        if len(bbox) == 4:
            self.top = bbox[1]
            self.bottom = bbox[3]
            self.left = bbox[0]
            self.right = bbox[2]
        else:
            self.top = min(bbox[1],bbox[3],bbox[5],bbox[7])
            self.bottom = max(bbox[1],bbox[3],bbox[5],bbox[7])
            self.left = min(bbox[0],bbox[6],bbox[2],bbox[4])
            self.right = max(bbox[0],bbox[6],bbox[2],bbox[4])
            self.bbox = [self.left,self.top,self.right,self.bottom]


class Layout_Row(object):
    def __init__(self):
        self.print = []
        self.hand = []
        self.print_below = []
        self.top = 9999
        self.bottom = 0


    def add_print(self,print):
        self.print.append(print)
        if print.top<self.top:
            self.top = print.top

        if print.bottom>self.bottom:
            self.bottom = print.bottom

    def add_hand(self,hand):
        if hand.centre[1]>self.top and hand.centre[1]<self.bottom:
            self.hand.append(hand)
        else:
            self.print_below.append(hand)

    def find_bbox(self):
        self.min_top = 9999
        self.min_left = 9999
        self.max_bottom = 0
        self.max_right = 0
        self.all_cell = self.print+self.hand+self.print_below
        for cell in self.all_cell:
            if cell.top<self.min_top:
                self.min_top = cell.top
            if cell.left < self.min_left:
                self.min_left = cell.left
            if cell.bottom > self.max_bottom:
                self.max_bottom = cell.bottom
            if cell.right > self.max_right:
                self.max_right = cell.right

        self.bbox = [self.min_left,self.min_top,self.max_right,self.max_bottom]




def row_iou(row1,row2):
    max_top = max(row1[0],row2[0])
    min_bottom = min(row1[1],row2[1])

    if max_top>=min_bottom:
        return 0
    else:
        return (min_bottom-max_top)/(row1[1]-row1[0]+row2[1]-row2[0]+max_top-min_bottom)

def column_iou(column1,column2):
    max_left = max(column1[0],column2[0])
    min_right = min(column1[1],column1[1])

    if max_left>=min_right:
        return 0
    else:
        return (min_right-max_left)/(column1[1]-column1[0]+column2[1]-column2[0]+max_left-min_right)




class Layout_Column(object):
    def __init__(self):
        self.print = []
        self.hand = []
        self.print_right = []
        self.left = 9999
        self.right = 0

    def add_print(self, print):
        self.print.append(print)
        if print.left < self.left:
            self.left = print.left

        if print.right > self.right:
            self.right = print.right


class Layout_Analysis(object):
    def __init__(self):
        self.print_list = []
        self.hand_list = []
        self.rows = []
        self.columns = []


    def adjust_new_row(self):                      #判断是否要新添加一个row
        pass


    def row_add(self):                                  #iou版本
        for print in self.print_list:
            find = False
            if len(self.rows) == 0:
                layout_row = Layout_Row()
                layout_row.add_print(print)
                self.rows.append(layout_row)

            else:
                for layout_row in self.rows:
                    if row_iou((layout_row.top,layout_row.bottom),(print.top,print.bottom))>0:
                        layout_row.add_print(print)
                        find = True
                        break
                if find:
                    continue
                else:
                    new_row = Layout_Row()
                    new_row.add_print(print)
                    self.rows.append(new_row)

        def get_top(print):
            return print.top

        self.rows.sort(key=get_top)

        for hand in self.hand_list:
            for i,row in enumerate(self.rows):
                if i+1 == len(self.rows):
                    row.add_hand(hand)
                elif hand.centre[1]>row.top and hand.centre[1]<self.rows[i+1].top:
                    row.add_hand(hand)
                    break

    def column_add(self):                   #iou版本
        for print in self.print_list:
            find = False
            if len(self.columns) == 0:
                layout_column = Layout_Column()
                layout_column.add_print(print)
                self.columns.append(layout_column)

            else:
                for layout_column in self.columns:
                    if column_iou((layout_column.left, layout_column.right), (print.left, print.right)) > 0:
                        layout_column.add_print(print)
                        find = True
                        break
                if find:
                    continue
                else:
                    new_column = Layout_Column()
                    new_column.add_print(print)
                    self.columns.append(new_column)

        def get_top(print):
            return print.left

        self.columns.sort(key=get_top)



    def row_cell_num_seg(self):
        num_top = len(self.rows[0].print)
        seq_num = [0]

        for i,row in enumerate(self.rows):
            if len(row.print) != num_top:
                num_top = len(row.print)
                seq_num.append(i)

        return seq_num

    def row_cell_gap_seq(self):
        seq_num = []

        for i,row in enumerate(self.rows):
            pass

    def row_hand_seq(self):
        seq_disturb_num = []
        seq_vertical_num = []
        for i,row in enumerate(self.rows):
            if len(row.print)*2<=len(row.print_below):
                seq_vertical_num.append(i)
            elif len(row.hand) == 0:
                seq_disturb_num.append(i)
            row.find_bbox()

        return seq_disturb_num,seq_vertical_num





    # def row_analysis(self):














if __name__ == '__main__':


    xml_path = config.DATA_XML
    all_img = set_xml_data(xml_path)

    for img_result in tqdm(all_img):

    # img_result = all_img[0]

        layout_analysis = Layout_Analysis()

        for print in img_result.print_word:
            layout_cell = Layout_Cell(print.bbox,print.label,'print')
            layout_analysis.print_list.append(layout_cell)

        for hand in img_result.hand_word:
            layout_cell = Layout_Cell(hand.bbox, hand.label, 'hand')
            layout_analysis.hand_list.append(layout_cell)


        layout_analysis.row_add()
        layout_analysis.column_add()

        seq_disturb_num, seq_vertical_num = layout_analysis.row_hand_seq()

        img = img_result.img.copy()
        img2 = img_result.img.copy()

        thickness = int(img.shape[0]/1000)+1



        # for i,row in enumerate(layout_analysis.rows):
        #     if i in seq_vertical_num:
        #         cv2.line(img,(0,row.top),(img.shape[0]-1,row.top),color=(0, 0, 255),thickness=thickness)
        #         cv2.line(img, (0, row.bottom), (img.shape[0] - 1, row.bottom), color=(0, 0, 255), thickness=thickness)
        #     elif i in seq_disturb_num:
        #         cv2.line(img, (0, row.top), (img.shape[0] - 1, row.top), color=(255, 0, 0),thickness=thickness)
        #     else:
        #         cv2.line(img, (0, row.top), (img.shape[0] - 1, row.top), color=(0, 255, 0),thickness=thickness)
        #
        # for bbox in img_result.print_word:
        #     draw_bbox(bbox.bbox,img2,(255,0,0))
        #
        # for bbox in img_result.hand_word:
        #     draw_bbox(bbox.bbox,img2,(0,0,255))

        for i,row in enumerate(layout_analysis.rows):
            if i in seq_vertical_num:
                draw_bbox(row.bbox,img,(0,0,255))




        cv2.imwrite(img_result.img_path.replace('data/img','layout_analysis4'),img)
        # cv2.imwrite(img_result.img_path.replace('data/img', 'layout_analysis3').replace('.','_.'), img2)


