import algorithm
import paths

# Functions done on TXT files (srt and script) preparing for analysing
# algorithm.replace_tab_with_space(script_name)
algorithm.script_clean(paths.script_name, paths.spc_to_name, paths.spc_to_sent)
algorithm.clean_subtitles(paths.subs_name)
algorithm.creating_files_for_compare(paths.clean_subtitles, paths.clean_script, paths.spc_to_name)

# Functions done on CSV files or create a CSV file
# algorithm.csv_two_clks(paths.clean_script, paths.spc_to_name, paths.spc_to_sent)
# algorithm.extract_clks(paths.myCsv)
# algorithm.normal_clks(paths.clks)


# Functions used to analyze the data
# normal_critical_word_list = algorithm.derivative_and_find_index(paths.normal_clks, paths.num_of_max, paths.num_of_min)
# srt_sentence_index = algorithm.find_word_index_with_matching_sequences\
    # (normal_critical_word_list, paths.comparison_script, paths.comparison_srt_1)
# critical_ts_list = algorithm.find_ts_with_word_index(srt_sentence_index, paths.clean_subtitles)

# print("Found", len(critical_ts_list), "out of", len(normal_critical_word_list), paths.critical_moments)

# moments_found_script = len(normal_critical_word_list)
# moments_founds_srt = len(critical_ts_list)

algorithm.two_clks_from_srt(paths.clean_subtitles)
algorithm.normal_clks_srt(paths.srt_clks)
# TODO: Add find_timeStamp_from_srt()
# TODO: Add create_bookmark_from_srt_timestamps
algorithm.png_graph_srt(5, paths.normal_clks_srt, paths.movie_path)
algorithm.bookmark_png(algorithm.movie_length(paths.movie_path), paths.bookmarks, paths.movie_path)
algorithm.png_graph(5, paths.normal_clks, paths.movie_path)

normal_critical_word_list = algorithm.derivative_and_find_index(paths.normal_clks_srt, paths.num_of_max, paths.num_of_min)
srt_sentence_index = algorithm.find_word_index_for_srt_normal_clocks(normal_critical_word_list)
critical_ts_list = algorithm.find_ts_with_word_index(srt_sentence_index, paths.clean_subtitles)
print(critical_ts_list)

# algorithm.similarity_srt_script("runFiles/comparison_script.txt", "runFiles/comparison_srt_1.txt")
# algorithm.graph_plot("runFiles/n_clks.csv", paths.movie_name)
algorithm.graph_plot_inter("runFiles/n_clks_srt.csv", paths.movie_name)
algorithm.graph_plot("runFiles/n_clks_srt.csv", paths.movie_name)

# algorithm.bookmark_graph(algorithm.movie_length(paths.movie_path), "runFiles/bookmarks.txt", paths.movie_path)




