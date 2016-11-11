
from Tkinter import *
import tkFileDialog
import ImageChops
import math
import Image
import cv2
import numpy as np
from PIL import Image, ImageTk

##----------------------------------------------------------------------------------------------##
# The method used to calculate the root-mean-square difference between two images"
def rmsdiff(im1, im2):
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value*(idx**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return rms



##----------------------------------------------------------------------------------------------##
#This method is used to filter the big box with party names and symbols
def get_symbol_box():
    global _ , contours, hierarchy
    # Find contours in the image
    _, contours, hierarchy = cv2.findContours(binary_ballot_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # discard areas that are too large
        if h > 1500 and w > 700:
            continue

        # discard areas that are too small
        if h < 800 or w < 600:
            continue

        # draw rectangle around contour on original image
        #cv2.rectangle(original_ballot_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        symbol_area_image = original_ballot_image[y:y + h, x:x + w]
        cv2.imwrite("symbol_area_image.png", symbol_area_image)
        return "symbol_area_image.png"


##----------------------------------------------------------------------------------------------##
#crop the image and take only the right half of the symbol box
def crop_image(path):
    symbol_box = Image.open(path)
    width = symbol_box.size[0]
    height = symbol_box.size[1]
    cropped_symbol_box = symbol_box.crop((width / 2, 0, width, height))
    cropped_symbol_box.save("cropped_symbol_area.jpg")
    return "cropped_symbol_area.jpg"


##----------------------------------------------------------------------------------------------##
#seperate symbol and vote boxes from the cropped image
def seperate_symbol_boxes(crop_symbol_box_path):

    rows = 0
    # Creates a matrix for symols and their respective vote numbers
    Matrix = [[0 for x in range(0,2)] for y in range(0,11)]

    cropped_symbol_box = cv2.imread(crop_symbol_box_path, cv2.IMREAD_COLOR)
    cropped_symbol_gray = cv2.cvtColor(cropped_symbol_box, cv2.COLOR_BGR2GRAY)
    ret, cropped_symbol_binary = cv2.threshold(cropped_symbol_gray, 200, 255, cv2.THRESH_BINARY)

    _, contours_b, hierarchy_b = cv2.findContours(cropped_symbol_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # get contours
    # for each contour found, draw a rectangle around it on original image

    for contour_b in contours_b:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour_b)

        # discard areas that are too large
        if h > 250 and w > 800:
            continue

        # discard areas that are too small
        if h < 80 or w < 100:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(cropped_symbol_box, (x, y), (x + w, y + h), (255, 0, 255), 2)
        symbol_vote_boxes = cropped_symbol_box[y:y + h, x:x + w]
        Matrix[rows / 2][rows % 2] = symbol_vote_boxes;
        rows = rows + 1

    return Matrix



##----------------------------------------------------------------------------------------------##
def identify_vote_symbol(Matrix):
    global crossed_boxes
    crossed_boxes=0
    for rows in range(0, 11):
        vote_box = Matrix[rows][0]
        vote_box_gray = cv2.cvtColor(vote_box, cv2.COLOR_BGR2GRAY)
        ret, vote_box_binary = cv2.threshold(vote_box_gray, 150, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(vote_box_binary, 100, 250)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=10, maxLineGap=50)
        hough = np.zeros(vote_box.shape, np.uint8)

        num_of_lines= 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough, (x1, y1), (x2, y2), (255, 255, 255), 2)
            num_of_lines = num_of_lines + 1

        if (num_of_lines > 4):
            crossed_boxes = crossed_boxes + 1
            cv2.imwrite("voted_symbol.jpg", Matrix[rows][1])


##----------------------------------------------------------------------------------------------##
#validation method
def check_validity():
    global candidate_list
    number_list=[]
    flag = 0

    for val in list_num:
        if selected.__contains__(val):
            flag = 1

    for number in range(1, 41):
        if number == 26 or number == 27:
            break
        number_list.append(number)

    if crossed_boxes != 1 or (voted_numbers > 3) or (flag == 1) or has_matching_symbol != 1:
        return 0
    else:

        candidate_list= list(set(number_list) - set(list_not_voted))
        print candidate_list
        return 1



##----------------------------------------------------------------------------------------------##
#match the voted symbol with available templates to identify the voted party
def match_symbol():
    global matching_symbol, has_matching_symbol
    voted_symbol = cv2.imread('voted_symbol.jpg')

    for j in range(1, 12):
        template = cv2.imread('symbols\P' + str(j) + '.jpg')
        res = cv2.matchTemplate(voted_symbol, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.5:
            has_matching_symbol=1
            matching_symbol = template


##----------------------------------------------------------------------------------------------##
# This method is used to filter the big box with candidate numbers
def get_number_box():
    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # discard areas that are too large
        if h > 300 and w > 200:
            continue

        # discard areas that are too small
        if h < 200 or w < 500:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(original_ballot_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        number_area_image = original_ballot_image[y:y + h, x:x + w]
        cv2.imwrite("number_area_image.jpg", number_area_image)
        return number_area_image


##----------------------------------------------------------------------------------------------##
#this method extract number boxes and find out which are crossed
def seperate_number_boxes(number_area_image):

    global voted_numbers,flag,list_num,selected,list_not_voted
    voted_numbers = 0
    gray_number_area = cv2.cvtColor(number_area_image, cv2.COLOR_BGR2GRAY)
    ret, binary_num_area = cv2.threshold(gray_number_area, 150, 255, cv2.THRESH_BINARY)

    # get contours
    _, contours_s, hierarchy_s = cv2.findContours(binary_num_area, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    number_box_list= [];
    num = 0;
    selected = []
    list_not_voted=[]
    list_num=[]

    # for each contour found, draw a rectangle around it on original image
    for contour_s in contours_s:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour_s)

        # discard areas that are too large
        if h > 1500 and w > 700:
            continue

        # discard areas that are too small
        if h < 40 or w < 40:
            continue

        num = num + 1;
        # draw rectangle around contour on original image
        cv2.rectangle(number_area_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        small_number_box = number_area_image[y:y + h, x:x + w]
        cv2.imwrite("small_box" + str(num) + ".jpg", small_number_box)
        number_box_list.append(small_number_box);

        gray = cv2.cvtColor(small_number_box, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 250)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=10, maxLineGap=50)
        hough = np.zeros(small_number_box.shape, np.uint8)

        number = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough, (x1, y1), (x2, y2), (255, 255, 255), 2)
            number = number + 1

    #if the box is crossed, then number of lines are >9
        if (number > 9):
            cv2.imwrite('crossed' + str(num) + '.png', small_number_box)
            voted_numbers = voted_numbers + 1
            selected.append(num)

        for j in range(1, 41):
            #since numbers 26 and 27 templates are missing, ignore them
            if j == 26 or j == 27:
                continue

            #match with the available number templates
            number_templates = Image.open('numbers\\' + str(j) + '.jpg')
            number_image = Image.open('small_box' + str(num) + '.jpg')  # trainImage
            match_score = rmsdiff(number_templates, number_image)
            if match_score < 580:
                list_not_voted.append(j)
                list_num.append(num)

##------------------------------------------DISPLAY RESULTS----------------------------------------------------##
def display_results(lbl,lbl2,lbl3):

    var1 = StringVar()
    label = Label(root, textvariable=var1, relief=RAISED)
    var1.set(lbl)
    label.pack()

    var4 = StringVar()
    label = Label(root, textvariable=var4, relief=RAISED)
    var4.set("Voted Party")
    label.pack()

    image = Image.open("voted_symbol.jpg")
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo
    label.pack()

    var2 = StringVar()
    label = Label(root, textvariable=var2, relief=RAISED)
    var2.set(lbl2)
    label.pack()

    var3 = StringVar()
    label = Label(root, textvariable=var3, relief=RAISED)
    var3.set(lbl3)
    label.pack()





##-----------------------------------------GUI BUTTON CLICK METHODS---------------------------------------##

#The method called upon clicking the select image button
def select_image():
    # Open the file locator and get the input image path
    global path
    path = tkFileDialog.askopenfilename()



#The method called upon clicking the calculate vote button
def calculate_vote():
    # global variables
    global original_ballot_image, gray_ballot_image, binary_ballot_image
    #load the original ballot image
    original_ballot_image=cv2.imread(path,cv2.IMREAD_COLOR)
    #convert the RGB image to gray scale
    gray_ballot_image=cv2.cvtColor(original_ballot_image, cv2.COLOR_BGR2GRAY)
    #convert the gray scale image to binary with threshold 150
    ret,binary_ballot_image= cv2.threshold(gray_ballot_image ,150,255,cv2.THRESH_BINARY)

    symbol_box_path=get_symbol_box()
    crop_symbol_box_path=crop_image(symbol_box_path)
    Matrix=seperate_symbol_boxes(crop_symbol_box_path)
    identify_vote_symbol(Matrix)
    match_symbol()

    number_area_image = get_number_box()
    seperate_number_boxes(number_area_image)

    validity=check_validity()
    if validity==0:
        lbl="Invalid vote!"
        lbl2=""
        lbl3=""
    else:
        lbl="Valid vote!"
        lbl2="Candidates who got votes"
        lbl3=candidate_list

    display_results(lbl,lbl2,lbl3)



##----------------------------------------GUI----------------------------------------------------##
root = Tk()
panelA = None
panelB = None

btn2 = Button(root,text='Find the vote',command = calculate_vote)
btn2.pack(side='bottom',fill='both',expand='yes',padx='10',pady='10')
btn1 = Button(root,text='Select Image',command = select_image)
btn1.pack(side='bottom',fill='both',expand='yes',padx='10',pady='10')


# w = Scale(root,from_=0,to_=100,orient=HORIZONTAL)
# w.pack(side='top')

root.mainloop()

# cv2.waitKey(0)