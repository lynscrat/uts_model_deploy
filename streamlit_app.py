import streamlit as st
import numpy as np
import pickle
import pandas as pd

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

def main():
    st.title('Hotel Booking Cancellation Prediction')

    # Fitur numerik
    no_of_adults = st.number_input('Number of Adults', min_value=1, value=2)
    no_of_children = st.number_input('Number of Children', min_value=0, value=0)
    no_of_weekend_nights = st.number_input('Weekend Nights', min_value=0, value=1)
    no_of_week_nights = st.number_input('Week Nights', min_value=0, value=1)
    required_car_parking_space = st.selectbox('Car Parking Space Required? 0 = No, 1 = Yes', [0, 1])
    lead_time = st.number_input('Lead Time (days)', min_value=0, value=30)
    avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, value=0.0)
    no_of_special_requests = st.number_input('Special Request', min_value = 0, value = 0)
    no_of_previous_cancellations = st.number_input('Previous Cancellation', min_value = 0, value = 0)
    no_of_previous_bookings_not_canceled = st.number_input('Previous Booking Not Canceled', min_value = 0, value = 0)
    arrival_year = st.selectbox('Year', [2017, 2018])
    arrival_month = st.number_input('Arrival Month', min_value = 1, max_value = 12, value = 1)
    arrival_date = st.number_input('Arrival Date', min_value = 1, max_value = 31, value = 1)
    repeated_guest = st.selectbox('Repeated Guest? 0 = No, 1 = Yes', [0, 1])



    # Fitur kategorikal
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3',
                                                             'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    market_segment_type = st.selectbox('Market Segment Type', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])

    if st.button('Predict Cancellation'):
        features = {
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': required_car_parking_space,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'arrival_year': arrival_year,  
            'arrival_month': arrival_month,
            'arrival_date': arrival_date,
            'market_segment_type': market_segment_type,
            'repeated_guest': repeated_guest,  
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests,  
            'no_of_previous_cancellations': no_of_previous_cancellations,  
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled 
        }

        # Buat DataFrame dan urutkan kolom sesuai dengan yang diharapkan model
        df_input = pd.DataFrame([features])
        
        # Urutkan kolom sesuai urutan yang diharapkan oleh model
        expected_columns = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                    'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
                    'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
                    'market_segment_type', 'repeated_guest',
                    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                    'avg_price_per_room', 'no_of_special_requests']
        df_input = df_input[expected_columns]

        # Encode input menggunakan encoder
        df_input = df_input.replace(encoder)

        # Prediksi
        prediction = model.predict(df_input)[0]

        # Tampilkan hasil
        if prediction == 1:
            st.error("Booking likely to be canceled ❌")
        else:
            st.success("Booking likely to be completed ✅")

if __name__ == '__main__':
    main()
