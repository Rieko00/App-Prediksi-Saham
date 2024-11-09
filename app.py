import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import base64
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler


with st.sidebar:
    selected = option_menu(
        menu_title="MENU",
        options=["HOME", "DATASET", "MODELING", "PREDIKSI"],
        icons=["house", "table", "sliders", "graph-up"],  # add the icons
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

if selected == 'HOME':
    st.write("# HOME")
    st.markdown(
        '<p style="text-align: justify;">'
        'Investasi saham telah menjadi salah satu instrumen yang populer di kalangan masyarakat, baik di Indonesia maupun secara global. Saham PT Bank Mandiri Tbk (BMRI) merupakan salah satu saham perbankan yang paling diminati oleh para investor karena kestabilannya dan prospek pertumbuhannya yang positif dalam jangka panjang. Namun, fluktuasi harga saham yang tidak dapat diprediksi menjadi tantangan bagi para investor, terutama yang menggunakan strategi jangka pendek. Dengan semakin berkembangnya teknologi dan ketersediaan data, metode prediksi harga saham berbasis sains data semakin relevan untuk memberikan panduan yang lebih baik dalam pengambilan keputusan investasi.'
        '</p>',
        unsafe_allow_html=True
    )
    st.image("data/chart.webp")
    st.markdown(
        '<p style="text-align: justify;">'
        'Aplikasi ini dibuat untuk memprediksi harga saham Bank Mandiri (BMRI) menggunakan metode Bagging. '
        'Pada aplikasi ini terdapat beberapa halaman yang dapat diakses, yaitu:'
        '<ul>'
        '<li>Dataset : Berisi informasi tentang dataset yang digunakan.</li>'
        '<li>Modelling : Berisi informasi tentang hasil modelling yang dilakukan.</li>'
        '<li>Prediksi : Berisi informasi tentang hasil prediksi harga saham</li>'
        '</ul>'
        '</p>',
        unsafe_allow_html=True
    )

if selected == 'DATASET':
    st.write("# DATASET")
    st.write("Pada halaman ini akan berisi tentang Informasi Dataset yang digunakan.")
    dataset, visual = st.tabs(["Data", "Visualisasi",])

    with dataset:
        st.markdown(
            '<p style="text-align: justify;">'
            'Data Saham diambil dari data historis saham Bank Mandiri (BMRI). Pengamatan data data saham ini mencakup '
            '<span class="highlight">tanggal 01-01-2019 hingga 10-11-2024</span>. '
            'Data ini dikumpulkan dan dipublikasikan oleh website investing.com'
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
            'Grafik data box plot'  
            '</p>',
            unsafe_allow_html=True
        )
        st.image('data/bmri_outlier.png')

if selected == 'MODELING':
    st.write("## Modelling")
    lr, svm, rf = st.tabs(['Bagging Linear Regression','Bagging SVM','Bagging Random Forest'])

    with lr:
        prediksi_lr = pd.read_excel("data/predict_lr.xlsx")
        histori_lr = pd.read_excel("data/linear_history.xlsx")
        dfhistory_lr = pd.DataFrame(histori_lr)

        histori,visual = st.tabs(['Histori','Visual'])
        with histori:
            st.write('Skenario bagging linear regression')

            st.info("#### Histori dari skenario bagging linear regression")
            st.markdown(
                '<p style="text-align: justify;">'
                'Untuk histori pembuatan model skenario dapat dilihat dibawah ini. Record yang disimpan merupakan pengujian dari kombinasi parameter yang mencapai <span class="highlight"> 359 Iterasi.</span>'
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
            st.write(f"Nilai RMSE : {round(dfhistory_lr['RMSE'].min(), 7)}")

        with visual:
            st.info("##### Visualisasi grafik")
            st.markdown(
                '<p style="text-align: justify;">'
                'Hasil prediksi dalam pengujian yang didapatkan akan dituangkan dalam grafik dan tabel berikut.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.image("data/predict_lr.png")

            st.info("##### Tabel Aktual dan Prediksi")

            st.dataframe(prediksi_lr, use_container_width=True) 


    with svm:
        prediksi_svm = pd.read_excel("data/predict_svm.xlsx")
        history_svm = pd.read_excel("data/svm_history.xlsx")
        dfhistory_svm = pd.DataFrame(history_svm)

        histori,visual = st.tabs(['Histori','Visual'])
        with histori:
            st.write('Skenario bagging SVM')

            st.info("#### Histori dari skenario bagging SVM")
            st.markdown(
                '<p style="text-align: justify;">'
                'Untuk histori pembuatan model skenario dapat dilihat dibawah ini. Record yang disimpan merupakan pengujian dari kombinasi parameter yang mencapai <span class="highlight"> 191 Iterasi.</span>'
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
            st.write(f"Nilai RMSE : {round(dfhistory_svm['RMSE'].min(), 7)}")

        with visual:
            st.info("##### Visualisasi grafik")
            st.markdown(
                '<p style="text-align: justify;">'
                'Hasil prediksi dalam pengujian yang didapatkan akan dituangkan dalam grafik dan tabel berikut.</span>'
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
            st.write('Skenario Bagging Random Forest')

            st.info("#### Histori dari skenario Bagging Random Forest")
            st.markdown(
                '<p style="text-align: justify;">'
                'Untuk histori pembuatan model skenario dapat dilihat dibawah ini. Record yang disimpan merupakan pengujian dari kombinasi parameter yang mencapai <span class="highlight"> 575 Iterasi.</span>'
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
            st.write(f"Nilai RMSE : {round(dfhistory_rf['RMSE'].min(), 7)}")

        with visual:
            st.info("##### Visualisasi grafik")
            st.markdown(
                '<p style="text-align: justify;">'
                'Hasil prediksi dalam pengujian yang didapatkan akan dituangkan dalam grafik dan tabel berikut.</span>'
                '</p>',
                unsafe_allow_html=True
            )
            st.image("data/predict_rf.png")

            st.info("##### Tabel Aktual dan Prediksi")
            st.dataframe(prediksi_rf, use_container_width=True)

if selected == 'PREDIKSI':
    svm_model = load('data/model_SVM.pkl')

    st.write("# Prediksi Harga Saham")
    # st.warning("###### Remainder")
    st.write("Halaman ini anda akan melakukan prediksi harga saham BMRI dengan data masukan berupa data harga saham BMRI dari waktu sebelumnya")

    st.info("Masukkan Data harga saham BMRI")

    col1,col2 = st.columns(2)
    with col1:
        t1 = st.number_input("Masukkan harga saham saat ini")
    with col2:
        t2 = st.number_input("Masukkan harga saham 1 hari sebelumnya")
    t3 = st.number_input("Masukkan harga saham 2 hari sebelumnya")

    if st.button("Predict"):
        newdata = np.array([t1, t2, t3])
        scaler = MinMaxScaler()
        normnewdata = scaler.fit_transform(newdata.reshape(-1, 1))
        prediksi = svm_model.predict(normnewdata.reshape(1,-1))
        denormpredict = scaler.inverse_transform(prediksi.reshape(-1,1))

        percentage_change = ((denormpredict[0][0] - t1) / t1) * 100
        change_sign = '+' if percentage_change > 0 else ''

        st.write(f"Hasil peramalan saham besok: ")
        # st.success(f"{round(denormpredict[0,0],2)}%")
        st.success(f'Prediksi harga BMRI Besok: Rp {round(denormpredict[0,0],2)} ({change_sign}{percentage_change:.2f}%)')

        if st.button("Reset"):
            t1 = 0
            t2 = 0
            t3 = 0