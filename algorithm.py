import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
import os

from scipy.interpolate import interp1d
from difflib import SequenceMatcher
from string import punctuation
from heapq import nlargest
from heapq import nsmallest
from moviepy.editor import VideoFileClip

clk_1_size = 0
clk_2_size = 0
first_clk_2 = 0
SIZE_OF_INTER = 1
my_dpi = 96
MOD_VAL = 5


def ffmpeg_apply_graph(movie_name, output_name):

    string = """ffmpeg -i """ + movie_name +  """ -i graph.png -filter_complex "[1:v]format=argb,geq=r='r(0,0)':a='0.3*alpha(0,0)'[zork]; [0:v][zork]overlay=0:575" -pix_fmt yuv420p -c:a copy """ + output_name
    print(string)
    os.system(string)


def movie_length(path):
    clip = VideoFileClip(path)
    length = int(clip.duration / 60)
    return length


# This function receives a script and replaces any instance of TAB with space and removes redundant
# endlines from the file. new file is untabbed_script.txt
def replace_tab_with_space(path):
    untabbed = open("runFiles/untabbed_script.txt", "w")
    with open(path, "r") as f:
        line = f.readline()
        while line:
            new_line = line.replace("\t", "  ")
            untabbed.writelines(new_line)
            line = f.readline()


# Script clean receives a path to a script.txt file
# The function cleans the script of all words that are not - who spoke them or what they said
# e.g: background sounds or behavioral expressions
def script_clean(path, spc_to_speaker, spc_to_sent):
    clean_txt = open("runFiles/clean_script.txt", "w")
    with open(path, "r") as txt_file:
        line = txt_file.readline()
        while line:
            ws_count = len(line) - len(line.lstrip(' '))
            if ws_count == spc_to_sent or ws_count == spc_to_speaker:
                clean_txt.write(line)
            line = txt_file.readline()

    with open("runFiles/clean_script.txt", "r") as f:  # Cleaning the script from any instances of parentheses
        input = f.read()                      # using REGEX that goes through the entire file
        output = re.sub("[\(\[].[\s\S]*?[\)\]]", "", input)
        clean_txt1 = open("runFiles/clean_script.txt", "w")
        clean_txt1.write(output)


# csv_twoClk receives the clean_script.txt it then creates a new CSV file
# that contains two columns - Name of speaker and how many words they said in a sentence
# i.e: Shrek, 8
def csv_two_clks(path, ws_speaker, ws_sentence):
    csv_file = open('runFiles/mycsv.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Name', 'Num of words'])
    tmp_name = ""

    with open(path, "r") as txt_file:
        line = txt_file.readline()
        while line:
            if not line.strip():  # Checking if line is not empty
                line = txt_file.readline()
                continue
            ws_count = len(line) - len(line.lstrip(' '))
            if ws_count == ws_speaker:   # If line has spc_to_speaker spaces then it represents a speaker name
                tmp_name = line.strip()  # saving the last speakers name so that we can insert it to the CSV file
                num_of_words = 0
            if ws_count == ws_sentence:
                num_of_words = len((line.strip()).split())
                csv_writer.writerow([tmp_name, num_of_words])
            line = txt_file.readline()


# extract_clks: receives a regular csv file and creates a new csv file
# which its columns represent our 2 un-normalised clocks = clks.csv
def extract_clks(path):
    clks_csv = open('runFiles/clks.csv', 'w', newline='')
    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(clks_csv)

    csv_writer.writerow(['Clock 1', 'Clock 2'])

    name_tmp = ""
    name_count = -1
    words_count = 0

    next(csv_reader)  # This is used to skip the column titles

    for line in csv_reader:
        if name_count == -1:      # first 2 ifs are used to initialise the counters
            name_tmp = line[0]
            name_count += 1
        if words_count == 0:
            words_count = int(line[1])
            continue
        if line[0] == name_tmp:   # This means the speaker hasn't changed then we continue summing the words said
            words_count += int(line[1])
        else:
            csv_writer.writerow([name_count, words_count])  # Writing the data to the new CSV
            name_tmp = line[0]     # This is the next speakers name
            words_count += int(line[1])
            name_count += 1


# This function receives a path to the CSV file that contains the two clocks C1 & C2
# and creates a new CSV file that contains the normalized clocks: N(C1) & N(C2)
# and their difference in column 3 (Clock1 - Clock2) that will be needed later
def normal_clks(path):
    global clk_1_size
    global clk_2_size
    global first_clk_2
    n_clks_csv = open('runFiles/n_clks.csv', 'w', newline='')
    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(n_clks_csv)

    csv_writer.writerow(['Normal Clock 1', 'Normal Clock 2', 'Clk_1 - Clk_2'])

    for line in csv_reader:
        pass

    clk_1_size = int(line[0])
    clk_2_size = int(line[1])

    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)

    next(csv_reader)
    flag = 0
    for line in csv_reader:  # In this loop we normalize the clocks using the formula to make sure the clocks are now in
        if flag == 0:        # the interval of [0,1]
            first_clk_2 = int(line[1])
            flag = 1
        csv_writer.writerow([int(line[0])/clk_1_size, (int(line[1]) - first_clk_2)/(clk_2_size - first_clk_2),
                             (int(line[0])/clk_1_size) - ((int(line[1]) - first_clk_2)/(clk_2_size - first_clk_2))])


# Graph plot for N(Clk1) and N(Clk1-Clk2)
def graph_plot(path, movie_name):
    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)
    x = []
    y = []
    next(csv_reader)

    for line in csv_reader:
        x.append(float(line[0]))
        y.append(float(line[2]))

    y_max = max(y)
    y_min = min(y)

    plt.plot(x, y, color='blue', linestyle='-', linewidth=3,
             markerfacecolor='blue', markersize=12)
    plt.ylim(y_min-0.01, y_max+0.01)
    plt.xlim(0, 1)
    plt.xlabel('Speaker Change')
    plt.ylabel('DV')
    plt.title(movie_name)
    plt.savefig('runFiles/graph_plot_noisy.png')
    plt.show()



# Graph plot + interpolate for N(Clk1) and N(Clk1-Clk2)
def graph_plot_inter(path, movie_name):
    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)
    x = []
    y = []

    size = 0

    next(csv_reader)
    for line in csv_reader:
        x.append(float(line[0]))
        y.append(float(line[2]))
        size = size + 1

    x = np.linspace(0, 1, num=size, endpoint=True)
    f2 = interp1d(x, y, kind='cubic')

    xnew = np.linspace(0, 1, num=size/SIZE_OF_INTER, endpoint=True)
    plt.figure(figsize=(1080 / my_dpi, 92 / my_dpi), dpi=my_dpi)
    plt.plot(xnew, f2(xnew), '-')
    plt.axis('off')
    plt.savefig('runFiles/graph_plot_interpolation.png')
    plt.show()


# Bookmark png using PIL
def bookmark_png(movie_length, bm_path, movie):

    with open(bm_path, "r") as f:
        line = f.readline()
        ts_minute_list = []
        prev_appended = 0
        while line:
            hour = float(line[1])
            if float(line[3]) == 0:
                minute = float(line[4])
            else:
                minute = float(line[3:5])
            if float(line[6]) == 0:
                second = float(line[7])
            else:
                second = float(line[6:8])
            f_min = (hour*60) + minute + (second/60)
            if (f_min - prev_appended) >= 3:
                ts_minute_list.append(f_min)
                prev_appended = f_min

            line = f.readline()

    normal_stamps = []
    for i in ts_minute_list:
        normal_stamps.append(i/movie_length)

    cap = cv2.VideoCapture(movie)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape
    image = Image.new('RGB', (frame_w, int(frame_h * 0.05)), (190, 190, 190))  # Silver color
    image_draw = ImageDraw.Draw(image)
    image_draw.line([(0, (int(frame_h * 0.05)/2)), (frame_w, (int(frame_h * 0.05)/2))], fill=(0, 0, 0), width=5)

    y = (int(frame_h * 0.05)/2)
    for i in normal_stamps:
        x = int(frame_w * i)
        image_draw.ellipse((x-5, y-20, x+5, y+20), fill="black", outline="black")

    image.save('runFiles/action_bookmarks.png')


# Here we clean the subtitles of the movie from all parentheses using a REGEX
def clean_subtitles(path):
    with open(path, "r") as f:  # Cleaning the subtitles from any instances of parentheses
        input = f.read()        # using REGEX that goes through the entire file
        output = re.sub("[\(\[</].[\s\S]*?[\>\)\]]", "", input)
        clean_txt1 = open("runFiles/clean_subtitles.txt", "w")
        clean_txt1.write(output)


# Function for derivative of normal Clocks so we can find critical moments
# derivative[i] = (clk_1-clk_2)[i]-((clk_1-clk_2)[i+1]
# We also find the 3 largest and the 3 smallest derivatives for timestamps
def derivative_and_find_index(path, num_of_max, num_of_min):
    elements = []
    derivative_list = []

    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)

    next(csv_reader)

    for line in csv_reader:
        elements.append(float(line[2]))

    for x in range(len(elements)):
        try:
            deriv = elements[x] - elements[x+1]
            derivative_list.append(deriv)
        except IndexError:
            break

    min_max_index = []

    count = 0
    #  This finds the all the min/max points in the graph and adds them to min_max_index
    for k in derivative_list:
        try:
            if k < 0 and derivative_list[count+1] > 0:  # This is a minimum point
                min_max_index.append(count)
                count += 1

            if k > 0 and derivative_list[count+1] < 0:  # This is a maximum point
                min_max_index.append(count)
                count += 1
            else:
                count += 1
        except IndexError:
            break

    min_max_value = []

    #  This takes the min_max_index and finds their corresponding Y value
    for n in min_max_index:
        min_max_value.append(float(elements[n]))

    #  We choose the number of min and max points
    max_values = nlargest(num_of_max, min_max_value)
    min_values = nsmallest(num_of_min, min_max_value)

    normal_critical_word_index = []
    count = 0
    for i in elements:
        if i in max_values or i in min_values:
            normal_critical_word_index.append(count)
            count += 1
        else:
            count += 1

    normal_critical_word_list = []
    csv_file = open(path, 'r')
    reader = csv.reader(csv_file)

    count = 0
    for line in reader:
        if count in normal_critical_word_index:
            try:
                normal_critical_word_list.append(float(line[1]))
                count += 1
            except ValueError:
                continue
        else:
            count += 1
    return normal_critical_word_list


# This function receives the critical word list and searches for a match between the script
# and SRT using sequence matching of the words
# and returns a list containing the indexes of the words in the SRT file
def find_word_index_with_matching_sequences(normal_word_list, script_path, srt_path):
    global clk_2_size
    global first_clk_2
    actual_critical_words = []

    # This is how we extract the word number
    for word in normal_word_list:
        actual_critical_words.append(int((word * clk_2_size) + first_clk_2))

    script_seq_list = []  # List of lists

    with open(script_path, "r") as script:
        script_data = script.read().replace('\x00', '')
        splitted_script = script_data.split()

        for word in actual_critical_words:
            try:
                script_seq = [splitted_script[word-2] + " " + splitted_script[word-1] + " " + splitted_script[word]
                              + " " + splitted_script[word+1] + " " + splitted_script[word+2]]
                script_seq_list.append(script_seq)
            except IndexError:
                continue

    with open(srt_path, "r") as srt:
        srt_data = srt.read()
        splitted_srt = srt_data.split()
        count_1 = 0
        srt_sentence_index = []

        for i in script_seq_list:
            split_seq = script_seq_list[count_1][0].split()
            count_1 += 1
            #print("Now looking for", split_seq)
            count_2 = 0

            for j in splitted_srt:
                if split_seq[0] == j:
                    count_2 += 1
                    if split_seq[1] == splitted_srt[count_2]:
                        count_2 += 1
                        if split_seq[2] == splitted_srt[count_2]:
                            count_2 += 1
                            if split_seq[3] == splitted_srt[count_2]:
                                count_2 += 1
                                if split_seq[4] == splitted_srt[count_2]:
                                    #print("the sentence is at word number", count_2+2, "in the SRT")
                                    srt_sentence_index.append(count_2+2)
                                    break
                                else:
                                    count_2 -= 3
                                    continue
                            else:
                                count_2 -= 2
                                continue
                        else:
                            count_2 -= 1
                            continue
                    else:
                        continue
                else:
                    count_2 += 1
                    continue
    #print(srt_sentence_index)
    return srt_sentence_index


# This function receives a list containing the indexes of the critical words
# in the SRT file, then returns their corresponding TimeStamps
def find_ts_with_word_index(srt_sentence_index, path):
    ts_list = []
    with open("runFiles/bookmarks.txt", "w") as w:
        with open(path, "r") as f:
            line = f.readline()
            word_count = 0
            tmp_ts = ''
            count = 0
            while line:
                if not line[0].isdigit():
                    word_count += len((line.strip()).split())
                    try:
                        if word_count >= int(srt_sentence_index[count]):
                            ts_list.append(tmp_ts)
                            w.write(tmp_ts)
                            w.write("\n")
                            count += 1
                    except IndexError:
                        break
                if line[0] == '0':
                    tmp_ts = line.replace('\n', '')
                line = f.readline()
            return ts_list


# This function turns the script and SRT files into long strings that then
# are used for comparing them and doing further analysing
def creating_files_for_compare(srt_path, script_path, space_to_name):
    with open(srt_path, "r") as f:
        tmp_txt = open("runFiles/tmp_srt.txt", "w")
        input = f.read()
        output = "".join(c for c in input if c not in punctuation)
        tmp_txt.write(output)

    with open('runFiles/tmp_srt.txt', "r") as f:
        tmp_txt = open("runFiles/comparison_srt.txt", "w")
        line = f.readline()
        while line:
            if not line[0].isdigit():
                line = line.replace("\n", " ")
                tmp_txt.write(line)
            line = f.readline()

    with open("runFiles/comparison_srt.txt", "r") as f:
        tmp_txt = open("runFiles/comparison_srt_1.txt", "w")
        line = f.readline()
        while line:
            line = re.sub("\s\s+", " ", line)
            tmp_txt.write(line)
            line = f.readline()

    with open(script_path, "r") as f:
        tmp_txt = open("runFiles/tmp_script.txt", "w")
        input = f.read()
        output = "".join(c for c in input if c not in punctuation)
        tmp_txt.write(output)

    with open('runFiles/tmp_script.txt', "r") as f:
        tmp_txt = open("runFiles/comparison_script.txt", "w")
        line = f.readline()
        while line:
            ws_count = len(line) - len(line.lstrip(' '))
            if ws_count != space_to_name:
                line = line.strip(" ")
                line = line.replace("\n"," ")
                line = re.sub("\s\s+", " ", line)
                tmp_txt.write(line)
            line = f.readline()


# Checking the similarity between the SRT file and script file
def similarity_srt_script(str1, str2):
    with open(str1, "r") as s1:
        with open(str2, "r") as s2:
            str1 = s1.read()
            str2 = s2.read()
            print("Similarity between script and subtitle is", SequenceMatcher(None, str1, str2).ratio())


# Apply bookmarks to video
def apply_overlay_cv(movie_path):

    cap = cv2.VideoCapture(movie_path)

    logo = cv2.imread("runFiles/bookmarks.png", -1)

    watermark = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
    watermark_h, watermark_w, watermark_c = watermark.shape

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    frame_h, frame_w, frame_c = frame.shape

    out = cv2.VideoWriter('outputMovie.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 24, (frame_w, frame_h))

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        frame_h, frame_w, frame_c = frame.shape

        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                overlay[i + int(frame_h * 0.9), j] = watermark[i, j]

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        out.write(frame)

        # cv2.imshow('frame', frame)

        print("Frame number " + str(count) + " has been writen")

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count == 10000:
            break

    cap.release()
    out.release()


# Create a png graph using the Two Clocks CSV file
def png_graph(mod_val, clks_path, movie_path):

    csv_file = open(clks_path, 'r')
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    y_axis = []
    x_axis = []
    count = 0
    for line in csv_reader:
        if count % mod_val == 0:  # Every 5th element from the CSV file
            y_axis.append(float(line[2]))
            x_axis.append(float(line[0]))
        count += 1

    max_y = max(y_axis)
    abs_min_y = abs(min(y_axis))

    cap = cv2.VideoCapture(movie_path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape
    image_height = int(frame_h * 0.2)
    image = Image.new('RGB', (frame_w, image_height), (190, 190, 190))  # Silver color background
    image_draw = ImageDraw.Draw(image)

    for i in range(0, len(y_axis)):
        y_axis[i] = float(y_axis[i] + abs_min_y)

    max_y = max(y_axis)

    for i in range(0, len(y_axis)):
        y_axis[i] = float(y_axis[i] / max_y)
    # print(y_axis)

    prev_x = 0
    prev_y = image_height
    next_x = 0
    next_y = 0
    for i in range(0, len(x_axis)):
        next_x = int(frame_w * x_axis[i])
        next_y = int((1 - y_axis[i]) * float(image_height))
        image_draw.line([(prev_x, prev_y), (next_x, next_y)], fill=(0, 0, 0), width=2)
        prev_x = next_x
        prev_y = next_y

    image.save('runFiles/graph_for_watermarking.png')
    image.save('graph.png')


def two_clks_from_srt(path):
    with open(path, "r") as f:
        csv_file = open('runFiles/srt_clks.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Time', 'Num of words'])

        line = f.readline()
        time_to_csv = 0
        words_to_csv = 0
        while line:
            line = f.readline()
            if not line.strip():
                continue
            if line[0] == '0' and line[1].isdigit():
                csv_writer.writerow([time_to_csv, words_to_csv])
                tmp_time = line[0:8]
                sec = float(tmp_time[6:8])
                sec_to_min = sec / 60
                min = float(tmp_time[3:5])
                hour = float(tmp_time[1])
                hour_to_min = hour * 60
                time_to_csv = sec_to_min + min + hour_to_min
                line = f.readline()
            if not line[0].isdigit():
                words_to_csv += len((line.strip()).split())


def normal_clks_srt(path):
    global clk_1_size
    global clk_2_size
    global first_clk_2
    n_clks_csv = open('runFiles/n_clks_srt.csv', 'w', newline='')
    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(n_clks_csv)

    csv_writer.writerow(['Normal Clock 1', 'Normal Clock 2', 'Clk_1 - Clk_2'])

    for line in csv_reader:
        pass

    clk_1_size = int(float(line[0]))
    clk_2_size = int(float(line[1]))

    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)

    next(csv_reader)
    flag = 0
    for line in csv_reader:  # In this loop we normalize the clocks using the formula to make sure the clocks are now in
        if flag == 0:        # the interval of [0,1]
            first_clk_2 = int(line[1])
            flag = 1
        csv_writer.writerow([int(float(line[0]))/clk_1_size, (int(float(line[1])) - first_clk_2)/(clk_2_size - first_clk_2),
                             (int(float(line[0]))/clk_1_size) - ((int(float(line[1])) - first_clk_2)/(clk_2_size - first_clk_2))])


# Create a png graph using the Two Clocks CSV file
def png_graph_srt(mod_val, n_clks_path, movie_path):

    csv_file = open(n_clks_path, 'r')
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    y_axis = []
    x_axis = []
    count = 0
    for line in csv_reader:
        if count % mod_val == 0:  # Every 5th element from the CSV file
            y_axis.append(float(line[2]))
            x_axis.append(float(line[0]))
        count += 1

    max_y = max(y_axis)
    abs_min_y = abs(min(y_axis))

    cap = cv2.VideoCapture(movie_path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape
    image_height = int(frame_h * 0.2)
    image = Image.new('RGB', (frame_w, image_height), (190, 190, 190))  # Silver color background
    image_draw = ImageDraw.Draw(image)

    for i in range(0, len(y_axis)):
        y_axis[i] = float(y_axis[i] + abs_min_y)

    max_y = max(y_axis)

    for i in range(0, len(y_axis)):
        y_axis[i] = float(y_axis[i] / max_y)
    # print(y_axis)

    prev_x = 0
    prev_y = image_height
    next_x = 0
    next_y = 0
    for i in range(0, len(x_axis)):
        next_x = int(frame_w * x_axis[i])
        # image_draw.ellipse((x, y+1, x, y+5), fill="black", outline="black")
        next_y = int((1 - y_axis[i]) * float(image_height))
        image_draw.line([(prev_x, prev_y), (next_x, next_y)], fill=(0, 0, 0), width=2)
        prev_x = next_x
        prev_y = next_y

    image.save('runFiles/graph_for_watermarking.png')


def find_word_index_for_srt_normal_clocks(normal_word_list):
    global clk_2_size
    global first_clk_2
    actual_critical_words = []

    # This is how we extract the word number
    for word in normal_word_list:
        actual_critical_words.append(int((word * clk_2_size) + first_clk_2))
    return actual_critical_words


# Bookmark graph - using Matplotlib
def bookmark_graph(LENGTH_OF_MOVIE, bm_path, movie):

    cap = cv2.VideoCapture(movie)
    ret, frame = cap.read()
    frame_h, frame_w, frame_c = frame.shape

    with open(bm_path, "r") as f:
        line = f.readline()
        ts_minute_list = []
        prev_appended = 0
        while line:
            hour = float(line[1])
            if float(line[3]) == 0:
                minute = float(line[4])
            else:
                minute = float(line[3:5])
            if float(line[6]) == 0:
                second = float(line[7])
            else:
                second = float(line[6:8])
            f_min = (hour*60) + minute + (second/60)
            if (f_min - prev_appended) >= 3:
                ts_minute_list.append(f_min)
                prev_appended = f_min

            line = f.readline()

    normal_stamps = []
    for i in ts_minute_list:
        normal_stamps.append(i/LENGTH_OF_MOVIE)
    normal_stamps.append(1)
    normal_stamps.append(0)
    val = 0  # this is the value where you want the data to appear on the y-axis.
    plt.xlim(0, 1)
    plt.figure(figsize=(frame_w / my_dpi, (frame_h * 0.1) / my_dpi), dpi=my_dpi)
    plt.axis('off')
    plt.plot(normal_stamps, np.zeros_like(normal_stamps) + val, "*")
    plt.savefig('runFiles/action_bookmarks.png')
    plt.show()





