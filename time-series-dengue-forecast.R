library(fpp2)
library(tidyverse)
library(lubridate)
library(tseries)
library(TSstudio)
library(vars)
library(lmtest)
library(forecast)
library(ggplot2)

# DATA PROCESSING AND TRANSFORMATION

# Load data
data_df <- read.csv("Transformed_data.csv",
                    stringsAsFactors = FALSE)

# Data for dengue cases
# Declare time series object
dengue1 <- ts(data = data_df$dengue,
              start = decimal_date(ymd("2012-1-1")),
              frequency = 365/7)
# Checking for null values
sum(is.na(dengue1)) #0
# Plot the data
autoplot(dengue1)

# The data is not stationary, thus use BoxCox transformation with a lambda value
lambda1 <- BoxCox.lambda(dengue1)
dengue2 <- BoxCox(dengue1, lambda1)
autoplot(dengue2) # Plot BoxCox transformed data

# Number of differences required for a seasonally stationary
nsdiffs(dengue1) # 0
ndiffs(dengue1) # 0
nsdiffs(dengue2) # 0
ndiffs(dengue2) # 1
# After computing, 1 seasonal difference was required

# Thus, apply difference to the data using the diff R function
dengue3 <- diff(dengue2)
# The transformations led to having a stationary time series
autoplot(dengue3)

'Using mstl function to decompose a time series into seasonal, trend and remainder components
 Remove the trend component, using seasonal component only '
dengue4 <- seasadj(mstl(dengue2))
autoplot(dengue4)

# Rainfall data, declare a time series object
rainfall1 <- ts(data = data_df$weekly_rainfall,
                start = decimal_date(ymd("2012-1-1")),
                frequency = 365/7)
# Checking for null values
sum(is.na(rainfall1)) #0
autoplot(rainfall1)
tsdisplay(rainfall1) # Plot the time series

# Data is not stationary, thus use BoxCox transformation with a lambda value
lambda2 <- BoxCox.lambda(rainfall1)
rainfall2 <- BoxCox(rainfall1, lambda2)
autoplot(rainfall2) # Data is now stationary
tsdisplay(rainfall2)

# Checking whether difference is required or not
ndiffs(rainfall2)
nsdiffs(rainfall2)
# KPSS test to check for stationarity of a series around a deterministic trend
kpss.test(rainfall2)
# No difference needed

# Temperature data
temp1 <- ts(data = data_df$weekly_temp,
            start = decimal_date(ymd("2012-1-1")),
            frequency = 365/7)
sum(is.na(temp1)) # 0
autoplot(temp1)
tsdisplay(temp1)

# BoxCox transformation with a lambda value
lambda3 <- BoxCox.lambda(temp1)
temp2 <- BoxCox(temp1, lambda3)
autoplot(temp2)
tsdisplay(temp2)
# Not much different with the plot of temp1 raw data, so we can use temp1 data directly

# Checking whether difference is required or not
ndiffs(temp1) # 0
ndiffs(temp2) # 0
nsdiffs(temp1) # 0
nsdiffs(temp2) # 0

# Using nsdiffs() and ndiffs() on the seasonal component of Boxcox dengue dataset
nsdiffs(dengue4)
ndiffs(dengue4) # 1, so non-seasonal differencing is needed
dengue5 <- diff(dengue4)
autoplot(dengue5)
acf(dengue5) # Plot ACF
pacf(dengue5) # Plot PACF


# DATA MODELLING

# Train, test split
outofsampleperiod <- 105 # test set, 105 weeks between 29 Dec 2019 and 1 Jan 2022 (weekly frequency)
dengue2_split <- ts_split(dengue2, sample.out = outofsampleperiod) # BoxCox transformed
dengue3_split <- ts_split(dengue3, sample.out = outofsampleperiod) # BoxCox transformed & diff (stationary)
dengue4_split <- ts_split(dengue4, sample.out = outofsampleperiod) # BoxCox transformed & seasonal
dengue5_split <- ts_split(dengue5, sample.out = outofsampleperiod) # BoxCox transformed & seasonal & diff (stationary)
rainfall2_split <- ts_split(rainfall2, sample.out = outofsampleperiod)
temp1_split <- ts_split(temp1, sample.out = outofsampleperiod)

# In-sample forecast, only using training set

# SARIMA model uses the stationary seasonal component of dengue data (dengue5)
sarima_auto <- auto.arima(dengue5_split$train)
sarima_auto_forecast <- forecast(sarima_auto)
accuracy(sarima_auto_forecast)
checkresiduals(sarima_auto_forecast)
# There is time series information left within the residuals, thus it isn't fit

# ARIMAX forecast for dengue cases,
# with rainfall variable
arimax_rain <- auto.arima(dengue2_split$train,
                          xreg=rainfall2_split$train)
arimax_rain_forecast <- forecast(arimax_rain,
                                 xreg=rainfall2_split$train)
accuracy(arimax_rain_forecast)
checkresiduals(arimax_rain_forecast)

# with temperature variable
arimax_temp <- auto.arima(dengue2_split$train,
                          xreg=temp1_split$train)
arimax_temp_forecast <- forecast(arimax_temp,
                                 xreg = temp1_split$train)
accuracy(arimax_temp_forecast)
checkresiduals(arimax_temp_forecast)

# with both variables
arimax_model <- auto.arima(dengue2_split$train,
                           xreg = cbind(rainfall2_split$train,
                                        temp1_split$train))
arimax_model_forecast <- forecast(arimax_model,
                                  xreg = cbind(rainfall2_split$train,
                                               temp1_split$train))
accuracy(arimax_model_forecast)
checkresiduals(arimax_model_forecast)

# S-ARIMA-X forecast for dengue cases,
# with rainfall variable
# cannot use stationary data here as xreg presents (dengue3 vs dengue5)
sarimax_rain <- auto.arima(dengue4_split$train,
                           xreg=rainfall2_split$train)
sarimax_rain_forecast <- forecast(sarimax_rain,
                                  xreg=rainfall2_split$train)
accuracy(sarimax_rain_forecast)
checkresiduals(sarimax_rain_forecast)

# with temp variable
sarimax_temp <- auto.arima(dengue4_split$train,
                           xreg=temp1_split$train)
sarimax_temp_forecast <- forecast(sarimax_dengue_temp,
                                  xreg=temp1_split$train)
accuracy(sarimax_temp_forecast)
checkresiduals(sarimax_temp_forecast)

# with both variables - auto
sarimax_auto <- auto.arima(dengue4_split$train,
                           xreg=cbind(rainfall2_split$train,
                                      temp1_split$train))
sarimax_auto_forecast <- forecast(sarimax_auto,
                                  xreg=cbind(rainfall2_split$train,
                                             temp1_split$train))
accuracy(sarimax_auto_forecast)
checkresiduals(sarimax_auto_forecast)

# with both variables - manual
sarimax_model <- arima(dengue4_split$train,
                       order = c(1, 1, 0),
                       seasonal = c(1, 0, 1))
sarimax_model_forecast <- forecast(sarimax_model)
accuracy(sarimax_model_forecast)
checkresiduals(sarimax_model_forecast)

# Out-of-sample forecast

# S-ARIMA-X Model - Seasonal Naive Method
# Seasonal Naive Method - forecast for weekly rainfall
seasonalNaive_rain <- auto.arima(rainfall2_split$train,
                                 xreg = temp1_split$train)
seasonalNaive_rain_forecast <- forecast(seasonalNaive_rain,
                                        xreg = temp1_split$test,
                                        h = 105)
autoplot(seasonalNaive_rain_forecast) +
  autolayer(rainfall2_split$test) +
  ggtitle("Forecasts from Seasonal naive method")
accuracy(seasonalNaive_rain_forecast, rainfall2_split$test)

# Seasonal Naive Method - forecast weekly temperature
seasonalNaive_temp <- auto.arima(temp1_split$train,
                                 xreg = rainfall2_split$train)
seasonalNaive_temp_forecast <- forecast(seasonalNaive_temp,
                                  xreg = rainfall2_split$test,
                                  h = 105)
autoplot(seasonalNaive_temp_forecast) +
  autolayer(temp1_split$test) +
  ggtitle("Forecasts from Seasonal naive method") # Rplot14
accuracy(seasonalNaive_temp_forecast, temp1_split$test)

# Seasonal Naive Method - forecast for dengue cases
seasonalNaive_dengue <- auto.arima(dengue4_split$train,
                                   xreg=cbind(seasonalNaive_rain_forecast$fitted,
                                              seasonalNaive_temp_forecast$fitted))
seasonalNaive_dengue_forecast <- forecast(seasonalNaive_dengue,
                                          xreg=cbind(rainfall2_split$test,
                                                     temp1_split$test),
                                          h = 105)
autoplot(seasonalNaive_dengue_forecast) +
  autolayer(dengue4_split$test)
accuracy(seasonalNaive_dengue_forecast, dengue4_split$test)

# S-ARIMA-X - Dynamic Harmonic Regression Model
# Dynamic Harmonic Regression Model - forecast for weekly rainfall
dynamicHarmonicReg_rain <- auto.arima(rainfall2_split$train,
                                      xreg = fourier(temp1_split$train, K = 3))

dynamicHarmonicReg_rain_forecast <- forecast(dynamicHarmonicReg_rain,
                                             xreg=fourier(temp1_split$test,
                                                     K = 3),
                                             h=105)
autoplot(dynamicHarmonicReg_rain_forecast) +
  autolayer(rainfall2_split$test)
accuracy(dynamicHarmonicReg_rain_forecast,
         rainfall2_split$test)

# Dynamic Harmonic Regression Model - forecast for weekly temp
dynamicHarmonicReg_temp <- auto.arima(temp1_split$train,
                                      xreg = fourier(rainfall2_split$train, K = 2))
dynamicHarmonicReg_temp_forecast <- forecast(dynamicHarmonicReg_temp,
                                             xreg=fourier(rainfall2_split$test, K = 2),
                                             h=105)
autoplot(dynamicHarmonicReg_temp_forecast) +
  autolayer(temp1_split$test)
accuracy(dynamicHarmonicReg_temp_forecast,
         temp1_split$test)

# Dynamic Harmonic Regression Model - forecast for dengue cases
dynamicHarmonicReg_dengue <- auto.arima(dengue4_split$train,
                                        xreg = cbind(dynamicHarmonicReg_rain_forecast$fitted,
                                                     dynamicHarmonicReg_temp_forecast$fitted))
dynamicHarmonicReg_dengue_forecast <- forecast(dynamicHarmonicReg_dengue,
                                               xreg = cbind(rainfall2_split$test,
                                                            temp1_split$test),
                                               h=105)
autoplot(dynamicHarmonicReg_dengue_forecast) +
  autolayer(dengue4_split$test)
accuracy(dynamicHarmonicReg_dengue_forecast,
         dengue4_split$test)

# feed forward NN model 
# ndiffs=1
# nsdiffs=0, no LT trend on yearly basis
# autoplot(forecast(nnetar(dengue2_split$train, p = 12, P = 1, period = 12), h = 120))

# feed forward NN model - without external reg 
nn_model_1 = nnetar(dengue2_split$train,
                    p = 12,
                    P = 3,
                    period = 12)

forecasted_nn_1 = forecast(nn_model_1,
                           PI=TRUE,
                           h = 105,
                           bootstrap=TRUE,
                           npaths=100) 

accuracy(forecasted_nn_1, dengue2_split$test)

resid_1 = nn_model_1$residuals
pvalue_1= Box.test(resid_1[105:length(resid_1)],
                   fitdf = 15,
                   type=c("Ljung-Box"),
                   lag=60)
pvalue_1

acf(resid_1[105:length(resid_1)])

autoplot(forecasted_nn_1) +
  autolayer(dengue2_split$test)

# feed forward NN model - with external reg
nn_model_2 = nnetar(dengue2_split$train,
                    p = 12,
                    P = 2,
                    period = 12,
                    xreg = cbind(rainfall2_split$train,
                                 temp1_split$train))
forecasted_nn_2 = forecast(nn_model_2,
                           PI=TRUE,
                           h = 105,
                           bootstrap=TRUE,
                           npaths=100,
                           xreg = cbind(rainfall2_split$train,
                                      temp1_split$train))

accuracy(forecasted_nn_2, dengue2_split$test)

resid_2 = nn_model_2$residuals
pvalue_2= Box.test(resid_2[105:length(resid_2)],
                   fitdf = 14,
                   type=c("Ljung-Box"),
                   lag=60)
pvalue_2

acf(resid_2[105:length(resid_2)])

autoplot(forecasted_nn_2) +
  autolayer(dengue2_split$test)

# with xreg, without xreg 
# P=1, 0.0343672 0.3908101 , 0.04197035 0.38029016 
# P=2, 0.02810417 0.22599061 , 0.03506159 0.34941198 --> BEST
# P=3, 0.02566569 0.30489360 , 0.02840284  0.25411577    
# P=4, 0.009270011 0.669463027 , 0.02007329 0.40035862 
# P=5, 0.005787487 0.442834531 , 0.008219474 0.472776302 
# P=6, 0.001021926 0.406976683 , 0.00143251 0.57169523 
