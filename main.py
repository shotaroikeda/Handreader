import math
import numpy
import matplotlib.pyplot as plt

# Define some constants:
image_w = 28
image_h = 28
num_data = 5000

laplace_const = 1

# Files Used
ascii_image = open('training/trainingimages')
training_data = open('training/traininglabels')

# final_processed is a tuple of (0, 1, 2 ,3, 4, 5, 6, 7, 8, 9)
# manual implementation to keep it editable to an extent
final_processed = [[], [], [], [], [], [], [], [], [], []]
count_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

##################################################################################
# Functions for training()
##################################################################################

def obtain_num_text(f):
    # NOTE: THIS IS ALSO USED TO READ THE TEST DATA
    written_num = []

    for line in range(0, image_w):
        # Create a line that can be eval()
        a_line = "["
        a_line += f.readline()
        a_line = a_line.replace('+', '1,')
        a_line = a_line.replace('#', '1,')
        a_line = a_line.replace(' ', '0,')
        a_line = a_line.rstrip('\n')
        a_line += ']'
        written_num.append(eval(a_line)) 

    return written_num

def obtain_result():
    num = eval(training_data.readline())
    count_nums[num]+=1
    return num

def add_data(data, result):
    for i in range(0, image_h):
        for j in range(0, image_w):
            if len(final_processed[result]) == 0:
                final_processed[result] = data
                break
            else:
                final_processed[result][i][j]+=data[i][j]
    
# def process_truth():
#     for i in range(0, 10):
#         divide = count_nums[i]
#         for j in range(0, image_h):
#             for k in range(0, image_w):
#                 final_processed[i][j][k] /= divide

##################################################################################
# Functions for testing
##################################################################################
num_testing = 1000
test_images= file('training/testimages')
test_data = file('training/testlabels')

total_wrong = 0

def adjust_laplace():
    print "Beginning to adjust for smoothing"
    for i in range(0, 10):
        for j in range(0, image_h):
            for k in range(0, image_w):
                final_processed[i][j][k] = (float(final_processed[i][j][k] + laplace_const) / float(count_nums[i] + laplace_const*10)) 
    print "Finished adjusting for Laplace smoothing"

def conclude(test_image):
    prob_map = []
    probability = 1L
    for i in range(0, 10):
        for j in range(0, image_h):
            for k in range(0, image_w):
                occur = 0
                if test_image[j][k] == 0:
                    occur = 1 - final_processed[i][j][k] 
                else:
                    occur = final_processed[i][j][k]
                probability += occur
        # print probability
        # raw_input('continue')
        prob_map.append(probability)
    return numpy.argmax(prob_map)

def check(num):
    answer = eval(test_data.readline().rstrip('\n'))
    if not (num == answer):
        global total_wrong
        total_wrong += 1
        print "num: %d\n answer: %d\n" % (num, answer)

##################################################################################
# Graphing heat maps
##################################################################################
def plot_num():
    column = numpy.arange(image_h, -1, -1)
    row = numpy.arange(image_w, -1, -1)
    for num_result in final_processed:
        data = numpy.array(num_result)
        plt.pcolor(data)
        plt.axis([0, image_h-1, 0, image_w-1])
        plt.xticks(np.arange(0, image_h+1)+0.5, column)
        plt.yticks(np.arange(0, image_w+1)+0.5, row)
        plt.show()
        

##################################################################################
# Training + Postprocessing
##################################################################################

def training():
    print "Training initial data with sample size %d" % (num_data)
    for num in range(0, num_data):
        data = obtain_num_text(ascii_image)
        result = obtain_result()
        add_data(data, result)
    # print "Created a heat map, now converting to probability map"
    # converts the heat map into a probability map
    # process_truth()
    # print "Finished the probability map"
    
    print "--------------------"
    print "success!"
    print count_nums
    print "--------------------"

    # clear some space up
    ascii_image.close()
    training_data.close()
    
def testing():
    print "Creating a probability map"
    adjust_laplace()
    
    print "Testing the provided data"
    for num in range(0, num_testing):
        test_image = obtain_num_text(test_images)
        result = conclude(test_image)
        check(result)
    print "Finished!"
    print "Got %d wrong" % (total_wrong)

def evaluation():
    pass

def print_final():
    for i in range(0, 10):
        print "-----\n"
        print "printing %d" % i
        for j in range(0, image_h):
            print final_processed[i][j]

if __name__ == '__main__':
    training()
    print_final()
    adjust_laplace()
    print_final()
    # testing()
