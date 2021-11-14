from support_functions import *

# Model and data parameters
T_AFTER_CUT = 15
BATCH_SIZE = 8
N = 16
T = 120
DIM_IN = 4
DIM_OUT = 3
EPOCHS = 100

data_path = '/Users/macbook/Desktop/UpWork/HeraSpace/Fish_Training_Data.csv'
df = get_data(data_path)
x_test_f, y_test_f = train_test_data(df=df, N=N, T=T)[1, 3]
inputs, outputs, inputs_test, outputs_test = model_input_otputs(df=df,
                                                                N=N,
                                                                T=T,
                                                                batch_size=BATCH_SIZE,
                                                                t_after_cut=T_AFTER_CUT)

model = lstm_model(batch_size=BATCH_SIZE, dim_in=DIM_IN, dim_out=DIM_OUT)

model, history = train_model(model=model,
                             epochs=EPOCHS,
                             batch_size=BATCH_SIZE,
                             N=N,
                             T=T,
                             t_after_cut=T_AFTER_CUT,
                             inputs=inputs,
                             outputs=outputs,
                             inputs_test=inputs_test,
                             outputs_test=outputs_test)

predicted_values = predict(model=model,
                           dim_in=DIM_IN,
                           dim_out=DIM_OUT,
                           x_test_f=x_test_f,
                           n=10)

with open('/Users/macbook/Downloads/HeraSpaceModel.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('/Users/macbook/Downloads/HeraSpacePredicted.pkl', 'wb') as f:
    pickle.dump(predicted_values, f)