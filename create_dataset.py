import cv2


def test1():
    path = 'lALPDgQ9qWSf04LNA_DNA2E_865_1008.png'

    rgb = cv2.imread(path)
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] == 221:
                rgb[i,j,:] = [255,0,0]



    rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    # cv2.imshow('a',rgb)
    cv2.imwrite('text.png',rgb)



def test2():
    path = 'lADPDgQ9qWTDRknNC7jND6A_4000_3000.jpg'

    rgb = cv2.imread(path)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            blue = rgb[i,j,:]
            a = blue[0]
            b = blue[1]
            c = blue[2]

            if a<100 and b<100 and c>150:

                rgb[i,j,:] = [255,0,0]


    cv2.imshow('a',rgb)
    cv2.waitKey()


if __name__ == '__main__':
    test2()