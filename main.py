import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import os
from copy import deepcopy as dc

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
temp_processed = []
count_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def reset_all():
    global temp_processed
    temp_processed = dc(final_processed)
    
    global test_images
    test_images = file('training/testimages')

    global outputfile
    outputfile = open('results/laplace_%d/results.txt' % (laplace_const), 'w')

    global test_data
    test_data = file('training/testlabels')

    global total_wrong
    total_wrong = 0

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
    no_num = False
    for i in range(0, image_h):
        if no_num:
            break
        for j in range(0, image_w):
            if len(final_processed[result]) == 0:
                final_processed[result] = data
                no_num = True
                break
            else:
                final_processed[result][i][j]+=data[i][j]

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
                global temp_processed
                temp_processed[i][j][k] = (float(temp_processed[i][j][k] + laplace_const) / float(count_nums[i] + laplace_const*10)) 
    print "Finished adjusting for Laplace smoothing"

def conclude(test_image):
    prob_map = []
    for i in range(0, 10):
        probability = 0
        for j in range(0, image_h):
            for k in range(0, image_w):
                occur = 0
                if test_image[j][k] == 0:
                    occur = math.log(1 - temp_processed[i][j][k])
                elif test_image[j][k] == 1:
                    occur = math.log(temp_processed[i][j][k])
                probability += occur
        prob_map.append(probability)
    return numpy.argmax(prob_map)

def check(num):
    answer = eval(test_data.readline().rstrip('\n'))
    if not (num == answer):
        global total_wrong
        total_wrong += 1

##################################################################################
# Graphing heat maps
##################################################################################
# Can only do these once

def plot_num():
    column = numpy.arange(0, image_h-1)
    row = numpy.arange(0, image_w-1)
    for num_result in temp_processed:
        if not os.path.exists("results/laplace_%d" % (laplace_const)):
            os.makedirs("results/laplace_%d" % (laplace_const))
        name = "results/laplace_%d/%d_heatmap.png" % (laplace_const, temp_processed.index(num_result))
        data = numpy.array(num_result)
        plt.pcolor(data)
        plt.gca().invert_yaxis()
        plt.axis([0, image_h-1, image_w-1, 0])
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        cb = plt.colorbar()
        cb.set_label("Values of probability")
        plt.savefig(name)
        plt.close()

##################################################################################
# Training + Postprocessing
##################################################################################

if not os.path.exists("results/laplace_%d" % (laplace_const)):
    os.makedirs("results/laplace_%d" % (laplace_const))
outputfile = open('results/laplace_%d/results.txt' % (laplace_const), 'w')

def training():
    print "Training initial data with sample size %d" % (num_data)
    for num in range(0, num_data):
        data = obtain_num_text(ascii_image)
        result = obtain_result()
        add_data(data, result)
    
    print "--------------------"
    print "success!"
    print count_nums
    print "--------------------"

    # clear some space up
    ascii_image.close()
    training_data.close()
    
def testing():
    print "Testing the provided data"
    msg = "Testing Laplace Constant of %d\n" % (laplace_const)
    outputfile.write(msg)
    for num in range(0, num_testing):
        test_image = obtain_num_text(test_images)
        result = conclude(test_image)
        check(result)
    print "Finished!"
    msg = "Got %d wrong\n\n" % (total_wrong)
    print msg
    outputfile.write(msg)
    test_images.close()
    test_data.close()
    outputfile.close()

if __name__ == '__main__':
    training()
    global temp_processed
    temp_processed = dc(final_processed)

    for r in range(0, 100):
        adjust_laplace()
        plot_num()
        testing()
        print "--------------------------\n"
        print "Reset ALL VARIABLES, INCREASE laplace by 1"
        reset_all()
        global laplace_const 
        laplace_const += 1
        print "Starting Next iteration..."
        print "-------------------------\n"


