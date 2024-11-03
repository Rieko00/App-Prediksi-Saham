import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import base64
import numpy as np
import joblib


with st.sidebar:
    selected = option_menu(
        menu_title="MENU",
        options=["HOME", "DATASET", "MODELING", "PREDIKSI"],
        icons=["house", "table", "sliders", "graph-up"],  # add the icons
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

if selected == 'DATASET':
    st.write("# DATASET")
    st.write("Pada halaman ini akan berisi tentang Informasi Dataset yang digunakan. Mulai dari banyak dataset, visual datasetm analisis korelasi ACF, dan analisis korelasi PACF.")
    dataset, visual = st.tabs(["Data", "Visualisasi",])

    with dataset:
        st.markdown(
            '<p style="text-align: justify;">'
            'Data Inflasi diambil dari data historis Web Bank Indonesia. Pengamatan data inflasi Indonesia ini mencakup '
            '<span class="highlight">tanggal 01-01-2003 hingga 01-05-2024</span>. '
            'Data ini dikumpulkan dan dipublikasikan oleh Bank Indonesia sebagai bagian dari upaya mereka untuk '
            'memantau dan mengelola stabilitas ekonomi negara.'
            '</p>',
            unsafe_allow_html=True
        )

        dataset = pd.read_excel("data/bmri_uni.xlsx")
        st.dataframe(dataset, use_container_width=True)
        dataset.drop(columns=['Tanggal'], inplace=True)
        st.info(f"Banyak Dataset : {len(dataset)}")
        st.warning(f'Informasi Dataset')
        st.dataframe(dataset.describe(), use_container_width=True)

    with visual:
        st.success("### Ploting Dataset")
        st.markdown(
            '<p style="text-align: justify;">'
            'Data ditampilkan dalam bentuk grafik seperti di bawah ini.'
            '</p>',
            unsafe_allow_html=True
        )
        st.image('data/bmri.png')
        st.markdown(
            '<p style="text-align: justify;">'
            'Grafik data terlihat berfluktuasi seiring waktu. dimana lonjakan Inflasi tertinggi terjadi pada tahun 2005 hingga 2006.'
            '</p>',
            unsafe_allow_html=True
        )
        st.image('data/bmri_outlier.png')
        st.markdown(
            '<p style="text-align: justify;">'
            'Grafik data terlihat berfluktuasi seiring waktu. dimana lonjakan Inflasi tertinggi terjadi pada tahun 2005 hingga 2006.'
            '</p>',
            unsafe_allow_html=True
        )

if selected == 'MODELING':
    st.write("## Modelling")
    lr, svm, rf = st.tabs(['Bagging Linear Regression','Bagging SVM','Bagging Random Forest'])

    with lr:
        prediksi_lr = pd.read_excel("data/predict_lr.xlsx")
        histori_lr = pd.read_excel("data/linear_history.xlsx")
        dfhistory_lr = pd.DataFrame(histori_lr)

        histori,visual = st.tabs(['Histori','Visual'])
        with histori:
            st.write('Pada skenario 1 digunakan pembagian dataset dengan rasio 80:20.')

            st.info("#### Histori dari skenario 1")
            st.markdown(
                '<p style="text-align: justify;">'
                'Untuk histori pembuatan model skenario 1 dapat dilihat dibawah ini. Record yang disimpan merupakan pengujian dari kombinasi parameter yang mencapai <span class="highlight"> 648 Iterasi.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.dataframe(dfhistory_lr, use_container_width=True)

            st.info("##### Parameter terbaik yaitu:")
            st.markdown(
                '<p style="text-align: justify;">'
                'Dari percobaan tersebut didapatkan parameter terbaik dengan tingkat kesalahan terendah sebagai berikut.'
                '</p>',
                unsafe_allow_html=True
            )
            idmin = dfhistory_lr['RMSE'].idxmin()
            st.dataframe(dfhistory_lr.iloc[idmin,1:7], use_container_width=True)

            st.info("##### Hasil Nilai Error Terendah:")
            st.write(f"Nilai RMSE : {round(dfhistory_lr['RMSE'].min(), 7)} %")

        with visual:
            st.info("##### Visualisasi grafik")
            st.markdown(
                '<p style="text-align: justify;">'
                'Hasil peramalan dalam pengujian yang didapatkan akan dituangkan dalam grafik dan tabel berikut.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.image("data/predict_lr.png")

            st.info("##### Tabel Aktual dan Prediksi")

            st.dataframe(prediksi_lr, use_container_width=True) 


    with svm:
        prediksi_svm = pd.read_excel("data/predict_svm.xlsx")
        history_svm = pd.read_excel("data/SVM_history.xlsx")
        dfhistory_svm = pd.DataFrame(history_svm)

        histori,visual = st.tabs(['Histori','Visual'])
        with histori:
            st.write('Pada skenario 1 digunakan pembagian dataset dengan rasio 80:20.')

            st.info("#### Histori dari skenario 1")
            st.markdown(
                '<p style="text-align: justify;">'
                'Untuk histori pembuatan model skenario 1 dapat dilihat dibawah ini. Record yang disimpan merupakan pengujian dari kombinasi parameter yang mencapai <span class="highlight"> 648 Iterasi.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.dataframe(dfhistory_svm, use_container_width=True)

            st.info("##### Parameter terbaik yaitu:")
            st.markdown(
                '<p style="text-align: justify;">'
                'Dari percobaan tersebut didapatkan parameter terbaik dengan tingkat kesalahan terendah sebagai berikut.'
                '</p>',
                unsafe_allow_html=True
            )
            idmin = dfhistory_svm['RMSE'].idxmin()
            st.dataframe(dfhistory_svm.iloc[idmin,0:10], use_container_width=True)

            st.info("##### Hasil Nilai Error Terendah:")
            st.write(f"Nilai RMSE : {round(dfhistory_svm['RMSE'].min(), 7)} %")

        with visual:
            st.info("##### Visualisasi grafik")
            st.markdown(
                '<p style="text-align: justify;">'
                'Hasil peramalan dalam pengujian yang didapatkan akan dituangkan dalam grafik dan tabel berikut.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.image("data/predict_svm.png")

            st.info("##### Tabel Aktual dan Prediksi")
            st.dataframe(prediksi_svm, use_container_width=True) 

    with rf:
        prediksi_rf = pd.read_excel("data/predict_rf.xlsx")
        history_rf = pd.read_excel("data/rf_history.xlsx")
        dfhistory_rf = pd.DataFrame(history_rf)

        histori,visual = st.tabs(['Histori','Visual'])
        with histori:
            st.write('Pada skenario 1 digunakan pembagian dataset dengan rasio 80:20.')

            st.info("#### Histori dari skenario 1")
            st.markdown(
                '<p style="text-align: justify;">'
                'Untuk histori pembuatan model skenario 1 dapat dilihat dibawah ini. Record yang disimpan merupakan pengujian dari kombinasi parameter yang mencapai <span class="highlight"> 648 Iterasi.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.dataframe(dfhistory_rf, use_container_width=True)

            st.info("##### Parameter terbaik yaitu:")
            st.markdown(
                '<p style="text-align: justify;">'
                'Dari percobaan tersebut didapatkan parameter terbaik dengan tingkat kesalahan terendah sebagai berikut.'
                '</p>',
                unsafe_allow_html=True
            )
            idmin = dfhistory_rf['RMSE'].idxmin()
            st.dataframe(dfhistory_rf.iloc[idmin,1:8], use_container_width=True)

            st.info("##### Hasil Nilai Error Terendah:")
            st.write(f"Nilai RMSE : {round(dfhistory_rf['RMSE'].min(), 7)} %")

        with visual:
            st.info("##### Visualisasi grafik")
            st.markdown(
                '<p style="text-align: justify;">'
                'Hasil peramalan dalam pengujian yang didapatkan akan dituangkan dalam grafik dan tabel berikut.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.image("data/predict_rf.png")

            st.info("##### Tabel Aktual dan Prediksi")
            st.dataframe(prediksi_rf, use_container_width=True)