from support_functions import *
import pickle

# Model and data parameters
T_AFTER_CUT = 15  # Time length after stateful cut
BATCH_SIZE = 8  # Data batch size
N = 16  # Independent sample size
T = 120  # Time length
DIM_IN = 4  # Input dimentions, number of variables passed to model
DIM_OUT = 3  # Output dimentions, number of predicted targets
EPOCHS = 100  # Number of model training runs

# Preparing data to model input
data_path = '/Users/macbook/Desktop/UpWork/HeraSpace/Fish_Training_Data.csv'
df = get_data(data_path)
x_train_f, x_test_f, y_train_f, y_test_f = train_test_data(df=df, N=N, T=T)
inputs, outputs, inputs_test, outputs_test = model_input_otputs(df=df,
                                                                N=N,
                                                                T=T,
                                                                batch_size=BATCH_SIZE,
                                                                t_after_cut=T_AFTER_CUT)
# Setting up the model
model = lstm_model(batch_size=BATCH_SIZE, dim_in=DIM_IN, dim_out=DIM_OUT)

# Model training
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

# Predicting targets
predicted_values = predict(model=model,
                           dim_in=DIM_IN,
                           dim_out=DIM_OUT,
                           x_test_f=x_test_f,
                           n=10)

# Saving trained model and predicted targets for further analysis
with open('/Users/macbook/Downloads/HeraSpaceModel.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('/Users/macbook/Downloads/HeraSpacePredicted.pkl', 'wb') as f:
    pickle.dump(predicted_values, f)
