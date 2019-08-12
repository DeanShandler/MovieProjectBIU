import algorithm
import paths
import os
from termcolor import colored


def begin():
    method = start()
    analysis(method)


def start():
    while True:
        try:
            text = """
                Please select analysis method: """
            print(colored(text, 'magenta'))
            message = """
                Press 1 - for script based analysis
                Press 2 - for Subtitles based analysis
                """
            method = input(colored(message, 'green'))
            if method.isdigit():
                if method == '1' or method == '2':
                    break
                else:
                    raise ValueError
            else:
                raise ValueError
        except ValueError:
            text = colored("""
                Invalid input. Please type 1 or 2""", 'red')
            print(text)
    return method


def analysis(method_nm):

    # Input movie path
    while True:
        movie_path = input("""                Enter path to movie: """)
        try:
            if not os.path.isfile(movie_path):
                raise FileNotFoundError
            else:
                break
        except FileNotFoundError:
            text = colored("File not found or directory invalid", 'red')
            print(text)

    movie_length = algorithm.movie_length(movie_path)

    # Script Analysis case
    if method_nm == '1':
        a = """
                                        ***Attention***
                Please make sure your script is symmetric:
                This means that the number of spaces or tabs to a name, sentence said and background description
                is constant and different between the 3 types within the file"""
        print(colored(a, 'cyan', attrs=['bold']))

        # Input Symmetric Y/N
        while True:
            yes_or_no = input(colored("""
                Is your script symmetric? [Y/N]. If not select 2 at the start """, 'green'))
            try:
                if yes_or_no == 'N' or yes_or_no == 'n':
                    begin()
                if yes_or_no == 'Y' or yes_or_no == 'y':
                    break
                else:
                    raise ValueError
            except ValueError:
                text = colored("Invalid input type [Y/N]", 'red')
                print(text)

        # Input path to script
        while True:
            script_name = input("""
                Enter path to script: """)
            try:
                if not os.path.isfile(script_name):
                    raise FileNotFoundError
                else:
                    break
            except FileNotFoundError:
                text = colored("File not found or directory invalid", 'red')
                print(text)

        # Input path to subs
        while True:
            subs_name = input("""
                Enter path to subtitle file: """)
            try:
                if not os.path.isfile(subs_name):
                    raise FileNotFoundError
                else:
                    break
            except FileNotFoundError:
                text = colored("File not found or directory invalid", 'red')
                print(text)

        # Input spaces or tabs
        while True:
            tabs_or_space = input("""
                Are the lines indented with spaces or tabs? [S/T] """)
            try:
                if tabs_or_space == 'T' or tabs_or_space == 't':
                    algorithm.replace_tab_with_space(script_name)
                    break
                if tabs_or_space == 'S' or tabs_or_space == 's':
                    break
                else:
                    raise ValueError
            except ValueError:
                text = "Invalid input type [S/T]"
                print(colored(text, 'red'))

        # Input number of spaces to name
        while True:
            num = input("""
                Enter number of tabs or spaces to a speaker name: """)
            try:
                spc_to_name = int(num)
                break
            except ValueError:
                text = colored("Invalid input", 'red')
                print(text)

        # Input number of spaces to sentence
        while True:
            num = input("""
                Enter number of tabs or spaces to a sentence spoken: """"")
            try:
                spc_to_sent = int(num)
                break
            except ValueError:
                text = colored("Invalid input", 'red')
                print(text)

        # Input num_of_max
        while True:
            num = input("""
                Enter number of max points wanted [0-40]: """)
            try:
                num_of_max = int(num)
                try:
                    if 0 <= num_of_max <= 40:
                        break
                    else:
                        raise ValueError
                except ValueError:
                    text = colored("Invalid input", 'red')
                    print(text)
                    continue
            except ValueError:
                text = colored("Invalid input", 'red')
                print(text)

        # Input num_of_min
        while True:
            num = input("""
                Enter number of min points wanted [0-40]: """)
            try:
                num_of_min = int(num)
                try:
                    if 0 <= num_of_min <= 40:
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print(colored("Invalid input", 'red'))
                    continue
            except ValueError:
                print(colored("Invalid input", 'red'))

        algorithm.script_clean(script_name, spc_to_name, spc_to_sent)
        print(colored("""                Script Cleaned""", 'green'))
        algorithm.clean_subtitles(subs_name)
        print(colored("""                Subtitles Cleaned""", 'green'))
        algorithm.creating_files_for_compare(paths.clean_subtitles, paths.clean_script, spc_to_name)
        print(colored("""                Created files ready for comparison""", 'green'))

        # Functions done on CSV files or create a CSV file
        algorithm.csv_two_clks(paths.clean_script, spc_to_name, spc_to_sent)
        print(colored("""                Created CSV file of the two clocks""", 'green'))
        algorithm.extract_clks(paths.myCsv)
        print(colored("""                Extracted Clocks""", 'green'))
        algorithm.normal_clks(paths.clks)
        print(colored("""                Normalized the clocks""", 'green'))

        # Functions used to analyze the data
        normal_critical_word_list = algorithm.derivative_and_find_index(paths.normal_clks, num_of_max, num_of_min)
        print(colored("""                Normalized critical word list created""", 'green'))
        srt_sentence_index = algorithm.find_word_index_with_matching_sequences\
            (normal_critical_word_list, paths.comparison_script, paths.comparison_srt_1)
        print(colored("""                Found word index in subtitles file""", 'green'))
        critical_ts_list = algorithm.find_ts_with_word_index(srt_sentence_index, paths.clean_subtitles)
        print("""                Found""", len(critical_ts_list), "out of", len(normal_critical_word_list))
        moments_found_script = len(normal_critical_word_list)
        moments_founds_srt = len(critical_ts_list)

        # Input modulo value
        while True:
            mod = input("""
                Enter modulo number for editing the graph [1-25]: """)
            try:
                mod_val = int(mod)
                if 1 <= mod_val <= 25:
                    break
                else:
                    raise ValueError
                continue
            except ValueError:
                print(colored("Invalid input", 'red'))
        # Creating graphs
        algorithm.bookmark_png(movie_length, paths.bookmarks, movie_path)
        algorithm.png_graph(mod_val, paths.normal_clks, movie_path)

        if moments_founds_srt * 4 < moments_found_script:
            print(colored("""Warning: Found less than 25% of matches between the script and subtitles try a different method!""", 'red'))

    # Subtitles Analysis case
    if method_nm == '2':

        # Input path to subs
        while True:
            subs_name = input("""
                Enter path to subtitle file: """)
            try:
                if not os.path.isfile(subs_name):
                    raise FileNotFoundError
                else:
                    break
            except FileNotFoundError:
                print(colored("File not found or directory invalid", 'red'))

        algorithm.clean_subtitles(subs_name)
        print(colored("""                Subtitles Cleaned""", 'green'))
        algorithm.two_clks_from_srt(paths.clean_subtitles)
        print(colored("""                Extracted Clocks""", 'green'))
        algorithm.normal_clks_srt(paths.srt_clks)
        print(colored("""                Normalized the clocks""", 'green'))

        # Input modulo value
        while True:
            mod = input("""             
                Enter modulo number for editing the graph [1-40]: """)
            try:
                mod_val = int(mod)
                if 1 <= mod_val <= 40:
                    break
                else:
                    raise ValueError
                continue
            except ValueError:
                print(colored("Invalid input", 'red'))

        algorithm.png_graph_srt(mod_val, paths.normal_clks_srt, movie_path)



    while True:
        ffmpeg_enable = input(colored("""
                Do you wish to apply overlay now? This may take a while [Y/N] """, 'green'))
        try:
            if ffmpeg_enable == 'N' or ffmpeg_enable == 'n':
                break
            if ffmpeg_enable == 'Y' or ffmpeg_enable == 'y':
                output = input(colored("""
                Please enter output name for the movie after overlay (no file type extension needed): """,
                                       'cyan'))
                out_name = output + ".mp4"

                string = """
                Starting to apply graph to movie - This might take a while - Please wait!
                                        *** TO QUIT PRESS Q ***
                        """
                print(colored(string, 'magenta', attrs=['bold']))
                algorithm.ffmpeg_apply_graph(movie_path, out_name)
                break
            else:
                raise ValueError
        except ValueError:
            text = colored("Invalid input type [Y/N]", 'red')
            print(text)



    # algorithm.similarity_srt_script("runFiles/comparison_script.txt", "runFiles/comparison_srt_1.txt")
    # algorithm.graph_plot("runFiles/n_clks.csv", movie_name)
    # algorithm.graph_plot_inter("runFiles/n_clks.csv", movie_name)
    # algorithm.bookmark_graph(movie_length, "runFiles/bookmarks.txt", movie_path)


text = """
                Welcome to the 'Two Clocks' movie analyser """
print(colored(text, 'blue', attrs=['bold']))
begin()







