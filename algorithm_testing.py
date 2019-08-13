import algorithm
import paths

val = input("Choose method: ")
if val == 'script':
    ############## FOR ANALYZING MOVIE SCRIPTS #################
    algorithm.replace_tab_with_space(paths.script_name)
    algorithm.script_clean(paths.script_name, paths.spc_to_name, paths.spc_to_sent)
    algorithm.clean_subtitles(paths.subs_name)
    algorithm.creating_files_for_compare(paths.clean_subtitles, paths.clean_script, paths.spc_to_name)

    algorithm.csv_two_clks(paths.clean_script, paths.spc_to_name, paths.spc_to_sent)
    algorithm.extract_clks(paths.myCsv)
    algorithm.normal_clks(paths.clks)

    normal_critical_word_list = algorithm.derivative_and_find_index(paths.normal_clks, paths.num_of_max, paths.num_of_min)
    srt_sentence_index = algorithm.find_word_index_with_matching_sequences(normal_critical_word_list, paths.comparison_script, paths.comparison_srt_1)
    critical_ts_list = algorithm.find_ts_with_word_index(srt_sentence_index, paths.clean_subtitles)

    print("Found", len(critical_ts_list), "out of", len(normal_critical_word_list))

    algorithm.png_graph(1, paths.normal_clks, paths.movie_path)

    inp = input("Do you have the movie path?")
    if inp == 'y' or inp == 'Y':
        movie_path = input("enter movie path: ")
        algorithm.bookmark_png(algorithm.movie_length(movie_path), paths.bookmarks, paths.movie_path)
    if inp == 'n' or inp == 'N':
        movie_length = int(input("enter length of movie in minutes: "))
        algorithm.bookmark_png(movie_length, paths.bookmarks, paths.movie_path)

    algorithm.graph_plot(paths.normal_clks, paths.movie_name)
    algorithm.graph_plot_inter(paths.normal_clks, paths.movie_name)

if val == 'subs':
    ########### FOR ANALYZING SUBTITLE FILE ONLY ############
    algorithm.clean_subtitles(paths.subs_name)
    algorithm.two_clks_from_srt(paths.clean_subtitles)
    algorithm.normal_clks_srt(paths.srt_clks)
    normal_critical_word_list = algorithm.derivative_and_find_index(paths.normal_clks_srt, paths.num_of_max,paths.num_of_min)
    srt_sentence_index = algorithm.find_word_index_for_srt_normal_clocks(normal_critical_word_list)
    critical_ts_list = algorithm.find_ts_with_word_index(srt_sentence_index, paths.clean_subtitles)

    print(critical_ts_list)

    algorithm.png_graph_srt(1, paths.normal_clks_srt, paths.movie_path)

    inp = input("Do you have the movie path?")
    if inp == 'y' or inp == 'Y':
        movie_path = input("enter movie path: ")
        algorithm.bookmark_png(algorithm.movie_length(movie_path), paths.bookmarks, paths.movie_path)
    if inp == 'n' or inp == 'N':
        movie_length = int(input("enter length of movie in minutes: "))
        algorithm.bookmark_png(movie_length, paths.bookmarks, paths.movie_path)

    algorithm.graph_plot(paths.normal_clks_srt, paths.movie_name)
    algorithm.graph_plot_inter(paths.normal_clks_srt, paths.movie_name)


########### UNUSED METHODS FOR TESTING ####################
# algorithm.similarity_srt_script("runFiles/comparison_script.txt", "runFiles/comparison_srt_1.txt")
# algorithm.bookmark_graph(algorithm.movie_length(paths.movie_path), "runFiles/bookmarks.txt", paths.movie_path)




