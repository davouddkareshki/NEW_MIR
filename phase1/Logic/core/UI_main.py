import streamlit as st
from streamlit_option_menu import option_menu 
import sys
from streamlit_lottie import st_lottie
import requests

sys.path.append("../")
import utils
import time
from enum import Enum
import random
from snippet import Snippet
from analyzer import LinkAnalyzer
from indexer.index_reader import Index_reader, Indexes

import pandas as pd 
import numpy as np

from typing import List, Optional, Any
import streamlit_shadcn_ui as ui
import streamlit_nested_layout 

snippet_obj = Snippet()

class movie_emojies(Enum):
    EMOJI1 = '📽'
    EMOJI2 = '🎬'
    EMOJI3 = '📺'
    EMOJI4 = '📹'
    
class color(Enum):
#    RED = "#FF0000"
    GREEN = "#00FF00"
#    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
#    WHITE = "#FFFFFF"
#    CYAN = "#00FFFF"
#    MAGENTA = "#FF00FF"
    GOLDEN = "F5BF03"
    YELLOW_OCHRE = "CB9D06"
    OLIVE = "#808000"
    ORANGE = "#FFA500"

class all_color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"
    GOLDEN = "F5BF03"
    YELLOW_OCHRE = "CB9D06"
    OLIVE = "#808000"
    ORANGE = "#FFA500"

def load_lottie_url(url) :
    r = requests.get(url) 
    if r.status_code != 200 : 
        return None 
    return r.json()
lottie_link = 'https://lottie.host/e924ae99-ec8a-4584-b54b-408aaf577005/7G1GWQFu1c.json'
lottie_link_2 = 'https://lottie.host/66cf8c44-4c59-46b7-8961-26c64e50fd9d/3OxguPAagm.json'

def get_top_x_movies_by_rank(x: int, results: list):
    #print('resterict :', x)
    path = "indexer/"  # Link to the index folder
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    if len(query) == 0 : return summary
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))

def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    num_filter_results,
    selected_menu,
):
    if len(search_term) > 0 and not search_button : return

    search_column,mid, ranking_column,_ = st.columns((3,1.5,2,0.5))
    with mid : 
        st_lottie(lottie_link_2, height = '20', key = 'coding')
    with search_column:
        corrected_query = utils.correct_text(search_term)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            ) 
            #print('------------daste khar-------------')
            #print(result)
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                if search_button : 
                    st.warning("No results found!")
                result = [('tt0137523',1), ('tt0167260',1), ('tt0167261',1), ('tt15239678',1), ('tt1375666',1), 
                            ('tt0060196',1), ('tt0120737',1), ('tt0133093',1), ('tt0468569',1), ('tt0172495',1)]

            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            
            if selected_menu == 'Advanced Search' :
                search_time(start_time, end_time)

        for i in range(len(result)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            with card[0].container():
                #st.title(f"[:orange[{info['title']}]]({info['URL']})")
                title_color = color.YELLOW.value
                title_html = f"""
                            <h1 style='margin-bottom: 0;'>
                                <a href='{info['URL']}' style='color:{title_color}; text-decoration: none;'>
                                    {info['title'] + random.choice(list(movie_emojies)).value}
                                </a>
                            </h1>
                            """
                
                opt_color = color.GOLDEN.value 
                st.markdown(title_html, unsafe_allow_html=True)
                out = 'Summary: '
                out =  out.replace(
                        out,
                        f"<b><font size='4' color={opt_color}>{out}</font></b>"

                        )
                out = out + get_summary_with_snippet(info, search_term)

                st.markdown(
                    out,
                    unsafe_allow_html=True,
                )


            with st.container():
                Directors = "Directors: "
                Directors =  Directors.replace(
                        Directors,
                        f"<b><font size='4' color={opt_color}>{Directors}</font></b>"

                        )
                Directors += "".join(Directors + ", " for Directors in info["directors"])
                Directors = Directors[0:len(Directors)-2]
                st.markdown(Directors, unsafe_allow_html=True,)
                
            with st.container():
                stars = "Stars: "
                stars =  stars.replace(
                        stars,
                        f"<b><font size='4' color={opt_color}>{stars}</font></b>"

                        )
                stars += "".join(stars + ", " for stars in info["stars"])
                stars = stars[0:len(stars)-2]
                st.markdown(stars, unsafe_allow_html=True,)
        
                topic_card = st.columns(1)
                with topic_card[0].container():
                    genres = "Genres: "
                    genres =  genres.replace(
                            genres,
                            f"<b><font size='4' color={opt_color}>{genres}</font></b>"
                            )
                    genres += "".join(genre + ", " for genre in info["genres"])
                    genres = genres[0:len(genres)-2]
                    st.markdown(genres, unsafe_allow_html=True,)

            with st.expander("Show More!"):
                for key in info : 
                    if key not in ['title','genres','directors','first_page_summary','id', 'stars', 'Image_URL', 'URL', 'related_links', 'synopsis'] :
                        if key == 'reviews' :                
                            data_output = info[key][0:5]
                            with st.expander('Reviews') :
                                for i,X in enumerate(data_output) :
                                    X = X[0]
                                    if type(X) == str :
                                        st.markdown(str(i+1) + ': ' + X,unsafe_allow_html=True) 

                        if type(info[key]) == str : 
                            out = key + ': '
                            out =  out.replace(
                                out,
                                f"<b><font size='4' color={opt_color}>{out}</font></b>"
                            )
                            out = out + info[key] 
                            st.markdown(out,unsafe_allow_html=True) 
                        else :
                            data_output = info[key][0:5]
                            gater = key 
                            gater = gater.replace('_',' ')
                            with st.expander(gater.capitalize()) :
                                for i,X in enumerate(data_output) :
                                    if type(X) == str :
                                        st.markdown(str(i+1) + ': ' + X,unsafe_allow_html=True) 
                                    #if type(X) != str :
                                    #    print(key) 

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )

    with ranking_column:
        movie_column,_, actor_column = ranking_column.columns((2.5,0.25,1))
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            #print(len(top_movies))
            #print(len(top_actors))

        with actor_column :
            if selected_menu == 'Advanced Search' : 
                st.markdown(f"**Top {num_filter_results} Actors :**")
            else : 
                st.markdown(f"**Related Actors :**")    
                
            print(top_actors)
            for actor in top_actors : 
                st.markdown(f":rainbow[{actor}]")
                #st.title(f":rainbow[{info['title']}]")

            st.divider()
        with movie_column:
            if selected_menu == 'Advanced Search' : 
                st.markdown(f"**Top {num_filter_results} Movies :**")
            else : 
                st.markdown(f"**Related Movies :**")
            
            for movie_id in top_movies:
                info = utils.get_movie_by_id(movie_id, utils.movies_dataset)
                
                with st.container():
                    title_color = color.GREEN.value
                    st.markdown(
                        f"""
                        <a href='{info['URL']}' style='color:{title_color}; text-decoration: none;'>
                            {info['title']}
                        </a>
                        """, 
                        unsafe_allow_html=True
                        )
                    st.markdown('',unsafe_allow_html=True)
            st.divider()


    pass

def main(): 
    st.set_page_config(page_title = 'IMDB', page_icon = ':🎥:', layout = 'wide')

    #st.subheader('Welcome To IMBD :')
    _,mid,_ = st.columns((1.5,1,1))
    with mid : 
        st.markdown(
            '<span style="color:green">Developed By: Davoud Kareshki💻</span>',
            unsafe_allow_html=True,
        )
    
    
    selected = option_menu(
        menu_title = 'Welcome To IMBD',
        menu_icon = 'camera-reels',
        options = ['Manual Search','Top Movies', 'Top Actors','Advanced Search'],
        icons = ['house','film','person','search'],
        orientation='horizontal', 
        styles = {
        'nav-link-selected':{'background-color': 'lightblue'}
        }
    )
    
    if selected == 'Top Movies' or selected == 'Top Actors': 
        result = [('tt0137523',1), ('tt0167260',1), ('tt0167261',1), ('tt15239678',1), ('tt1375666',1), 
                  ('tt0060196',1), ('tt0120737',1), ('tt0133093',1), ('tt0468569',1), ('tt0172495',1)]
        top_actors, top_movies = get_top_x_movies_by_rank(None, result)
        
        if selected == 'Top Actors' : 
            col1, col2, col3, col4, col5 = st.columns((1,1,1,1,1))
            for i,actor in enumerate(top_actors) : 
                out = str(i+1) + '- ' + actor
                if i % 3 == 0 : 
                    with col2 : 
                        ui.metric_card(
                            title = out 
                        ) 
                        #st.markdown(out,unsafe_allow_html=True)
                if i % 3 == 1 : 
                    with col3 : 
                        ui.metric_card(
                            title = out 
                        ) 
                        #st.markdown(out,unsafe_allow_html=True)
                if i % 3 == 2 : 
                    with col4 : 
                        ui.metric_card(
                            title = out 
                        ) 
                        #st.markdown(out,unsafe_allow_html=True)
            
        search_term = ''
        if selected == 'Top Movies' : 
            col1,col2 = st.columns((2,1))
            with col2 : 
                st_lottie(lottie_link_2, height = '1', key = 'movieing')
            with col1 : 
                for i in range(len(top_movies)):
                    card = st.columns([3, 1])
                    info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
                    with card[0].container():
                        title_color = color.YELLOW.value
                        title_html = f"""
                                    <h1 style='margin-bottom: 0;'>
                                        <a href='{info['URL']}' style='color:{title_color}; text-decoration: none;'>
                                            {info['title'] + random.choice(list(movie_emojies)).value}
                                        </a>
                                    </h1>
                                    """
                        
                        opt_color = color.GOLDEN.value 
                        st.markdown(title_html, unsafe_allow_html=True)
                        out = 'Summary: '
                        out =  out.replace(
                                out,
                                f"<b><font size='4' color={opt_color}>{out}</font></b>"

                                )
                        out = out + get_summary_with_snippet(info, search_term)

                        st.markdown(
                            out,
                            unsafe_allow_html=True,
                        )


                    with st.container():
                        Directors = "Directors: "
                        Directors =  Directors.replace(
                                Directors,
                                f"<b><font size='4' color={opt_color}>{Directors}</font></b>"

                                )
                        Directors += "".join(Directors + ", " for Directors in info["directors"])
                        Directors = Directors[0:len(Directors)-2]
                        st.markdown(Directors, unsafe_allow_html=True,)
                        
                    with st.container():
                        stars = "Stars: "
                        stars =  stars.replace(
                                stars,
                                f"<b><font size='4' color={opt_color}>{stars}</font></b>"

                                )
                        stars += "".join(stars + ", " for stars in info["stars"])
                        stars = stars[0:len(stars)-2]
                        st.markdown(stars, unsafe_allow_html=True,)
                
                        topic_card = st.columns(1)
                        with topic_card[0].container():
                            genres = "Genres: "
                            genres =  genres.replace(
                                    genres,
                                    f"<b><font size='4' color={opt_color}>{genres}</font></b>"

                                    )
                            genres += "".join(genre + ", " for genre in info["genres"])
                            genres = genres[0:len(genres)-2]
                            st.markdown(genres, unsafe_allow_html=True,)
                    
                    col_left, col_right = st.columns((2,1))
                    with col_left :
                        with st.expander("Show More!"):
                            for key in info : 
                                if key not in ['title','genres','directors','first_page_summary','id', 'stars', 'Image_URL', 'URL', 'related_links', 'synopsis'] :
                                    if key == 'reviews' :                
                                        data_output = info[key][0:5]
                                        with st.expander('Reviews') :
                                            for i,X in enumerate(data_output) :
                                                X = X[0]
                                                if type(X) == str :
                                                    st.markdown(str(i+1) + ': ' + X,unsafe_allow_html=True) 

                                    if type(info[key]) == str : 
                                        out = key + ': '
                                        out =  out.replace(
                                            out,
                                            f"<b><font size='4' color={opt_color}>{out}</font></b>"
                                        )
                                        out = out + info[key] 
                                        st.markdown(out,unsafe_allow_html=True) 
                                    else :
                                        data_output = info[key][0:5]
                                        gater = key 
                                        gater = gater.replace('_',' ')
                                        with st.expander(gater.capitalize()) :
                                            for i,X in enumerate(data_output) :
                                                if type(X) == str :
                                                    st.markdown(str(i+1) + ': ' + X,unsafe_allow_html=True) 
                    st.divider()
        pass 

    if selected == 'Manual Search' : 
        inputer,searcher,anime,_ = st.columns((2,3,1,1))
        with anime : 
            st_lottie(lottie_link, height = '1', key = 'movieing')
        with searcher :
            search_button = st.button("Search!")
        with inputer :
            search_term = st.text_input(label = 'nothing', label_visibility="collapsed")
            st.image('IMDB.png', width=150)
        
        search_max_num = 20
        search_weights = [1,1,1]
        search_method = 'OkapiBM25'
        unigram_smoothing = None
        alpha = None
        lamda = None 
        search_handling(
            search_button,
            search_term,
            search_max_num,
            search_weights,
            search_method,
            unigram_smoothing,
            alpha,
            lamda,
            num_filter_results=50,
            selected_menu=selected
        )
    
    if selected == 'Advanced Search' : 
        inputer,searcher,mid_points,anime = st.columns((2,3,3,1))
        with anime : 
            st_lottie(lottie_link, height = '20', key = 'movieing')
        
        with searcher :
            search_button = st.button("Search!")
        with inputer :
            search_term = st.text_input(label = 'nothing', label_visibility="collapsed")
            st.image('IMDB.png', width=150)
            
        with st.expander("Advanced Search"):
            search_max_num = st.number_input(
                "Maximum number of results", min_value=5, max_value=100, value=10, step=5
            )
            weight_stars = st.slider(
                "Weight of stars in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            weight_genres = st.slider(
                "Weight of genres in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            weight_summary = st.slider(
                "Weight of summary in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )
            slider_ = st.slider(f"Select the number of top movies to show", 1, 10, 5)

            search_weights = [weight_stars, weight_genres, weight_summary]
            search_method = st.selectbox(
                "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
            )

            unigram_smoothing = None
            alpha, lamda = None, None
            if search_method == "unigram":
                unigram_smoothing = st.selectbox(
                    "Smoothing method",
                    ("naive", "bayes", "mixture"),
                )
                if unigram_smoothing == "bayes":
                    alpha = st.slider(
                        "Alpha",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                    )
                if unigram_smoothing == "mixture":
                    alpha = st.slider(
                        "Alpha",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                    )
                    lamda = st.slider(
                        "Lambda",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                    )

        if "search_results" not in st.session_state:
            st.session_state["search_results"] = []
        search_handling(
            search_button,
            search_term,
            search_max_num,
            search_weights,
            search_method,
            unigram_smoothing,
            alpha,
            lamda,
            num_filter_results=slider_,
            selected_menu=selected
        )


if __name__ == "__main__":
    main()